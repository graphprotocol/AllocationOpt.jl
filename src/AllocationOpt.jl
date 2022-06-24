module AllocationOpt

using CSV
using GraphQLClient

export network_state, optimize_indexer, read_filterlists, push_allocations!, create_rules!

include("exceptions.jl")
include("domainmodel.jl")
include("query.jl")
include("service.jl")
include("ActionQueue.jl")
include("CLI.jl")

"""
    function network_state(id, whitelist, blacklist, pinnedlist, frozenlist, indexer_service_network_url)
    
# Arguments
- `id::AbstractString`: The id of the indexer to optimise.
- `whitelist::Vector{AbstractString}`: Subgraph deployment IPFS hashes included in this list will be considered for, but not guaranteed allocation.
- `blacklist::Vector{AbstractString}`: Subgraph deployment IPFS hashes included in this list will not be considered, and will be suggested to close if there's an existing allocation.
- `pinnedlist::Vector{AbstractString}`: Subgraph deployment IPFS hashes included in this list will be guaranteed allocation. Currently unsupported.
- `frozenlist::Vector{AbstractString}`: Subgraph deployment IPFS hashes included in this list will not be considered during optimisation. Any allocations you have on these subgraphs deployments will remain.
- `indexer_service_network_url::AbstractString`: The URL that exposes the indexer service's network endpoint. Must begin with http. Example: http://localhost:7600/network.
"""
function network_state(
    id::AbstractString,
    whitelist::AbstractVector{T},
    blacklist::AbstractVector{T},
    pinnedlist::AbstractVector{T},
    frozenlist::AbstractVector{T},
    indexer_service_network_url::AbstractString,
) where {T<:AbstractString}
    if !isempty(pinnedlist)
        @warn "pinnedlist is not currently optimised for."
    end
    userlists = vcat(whitelist, blacklist, pinnedlist, frozenlist)
    if !verify_ipfshashes(userlists)
        throw(BadSubgraphIpfsHashError())
    end

    # Construct whitelist and blacklist
    query_ipfshash_in = ipfshash_in(whitelist, pinnedlist)
    query_ipfshash_not_in = ipfshash_not_in(blacklist, frozenlist)

    # Get client
    client = Client(indexer_service_network_url)

    # Pull data from mainnet subgraph
    # TODO: Parameterise network_id
    network_id = 1
    repo = snapshot(client, query_ipfshash_in, query_ipfshash_not_in)
    network = networkparameters(client, network_id)

    # Handle frozenlist
    # Get indexer
    indexer, repo = detach_indexer(repo, id)

    # Reduce indexer stake by frozenlist
    fstake = frozen_stake(client, id, frozenlist)
    indexer = Indexer(indexer.id, indexer.stake - fstake, indexer.allocations, indexer.cut)
    return repo, indexer, network
end

"""
    function optimize_indexer(indexer, repo, minimum_allocation_amount, maximum_new_allocations)

# Arguments
- `indexer::Indexer`: The indexer being optimised.
- `repo::Repository`: Contains the current network state.
- `network::GraphNetworkParameters`: Contains the current network parameters.
- `minimum_allocation_amount::Real`: The minimum amount of GRT that you are willing to allocate to a subgraph.
- `maximum_new_allocations::Integer`: The maximum number of new allocations you would like the optimizer to open.
- `gas::Float64`: The gas in grt that the indexer will spend on the allocation transaction. We use this to
    calculate profit, but note that the assumption that this will be the price at the end of the allocation
    lifetime is probably bad. Gas is constantly changing.
- `allocation_lifetime::Integer`: The number of epochs for which these allocations would be open. An allocation earns indexing rewards upto 28 epochs.
- `pinnedlist::Vector{AbstractString}`: Subgraph deployment IPFS hashes included in this list will be guaranteed allocation. Pinnedlist allocations will be at least 0.1 GRT.
```
"""
function optimize_indexer(
    indexer::Indexer,
    repo::Repository,
    network::GraphNetworkParameters,
    minimum_allocation_amount::Real,
    maximum_new_allocations::Integer,
    gas::Float64,
    allocation_lifetime::Integer,
    pinnedlist::AbstractVector{<:AbstractString},
)
    
    max_allocation_lifetime = 28
    # TODO: Test
    if allocation_lifetime > max_allocation_lifetime &&
        allocation_lifetime < 0
        throw(InvalidAllocationLifetime())
    end

    # Optimise
    ωopt = optimize(indexer, repo)
    pinned_ixs::Vector{Int64} = findall(x -> x in pinnedlist, repo.subgraphs)
    ω = optimize(indexer, repo, ωopt, maximum_new_allocations, minimum_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)

    # Filter results with deployment IPFS hashes
    suggested_allocations = Dict(
        ipfshash(k) => v for (k, v) in zip(repo.subgraphs, ω) if v > 0.0
    )

    # @show ωopt
    @show nonzero(ωopt)
    @show nonzero(ω)
    return suggested_allocations
end

"""
    function read_filterlists(filepath)
        
# Arguments

- `filepath::AbstractString`: A path to the CSV file that contains whitelist, blacklist, pinnedlist, frozenlist as columns.
"""
function read_filterlists(filepath::AbstractString)
    # Read file
    path = abspath(filepath)
    csv = CSV.File(path; header=1, types=String)

    # Filter out missings from csv
    listtypes = [:whitelist, :blacklist, :pinnedlist, :frozenlist]
    cols = map(x -> collect(skipmissing(csv[x])), listtypes)

    return cols
end

"""
    function push_allocations!(indexer_id, management_server_url, proposed_allocations, whitelist, blacklist, pinnedlist, frozenlist)

# Arguments

- `indexer_id::AbstractString`: Indexer id
- `management_server_url::T`: Indexer management server url, in format similar to http://localhost:18000
- `proposed_allocations::Dict{T,<:Real}`: The set of allocation to open returned by optimize_indexer
- `whitelist::AbstractVector{T}`: Unused. Here for completeness.
- `blacklist::AbstractVector{T}`: Unused. Here for completeness.
- `pinnedlist::AbstractVector{T}`: Unused. Here for completeness.
- `frozenlist::AbstractVector{T}`: Make sure to not close these allocations as they are frozen.
"""
function push_allocations!(
    indexer_id::AbstractString,
    management_server_url::T,
    indexer_service_network_url::T,
    proposed_allocations::Dict{T,<:Real},
    whitelist::AbstractVector{T},
    blacklist::AbstractVector{T},
    pinnedlist::AbstractVector{T},
    frozenlist::AbstractVector{T},
) where {T<:AbstractString}
    actions = []

    # Query existing allocations that are not frozen
    existing_allocations = query_indexer_allocations(
        Client(indexer_service_network_url), indexer_id
    )
    existing_allocs::Dict{String,String} = Dict(
        ipfshash.(existing_allocations) .=> id.(existing_allocations)
    )
    existing_ipfs::Vector{String} = ipfshash.(existing_allocations)
    proposed_ipfs::Vector{String} = collect(keys(proposed_allocations))

    # Generate ActionQueue inputs
    reallocations, reallocate_ipfs = ActionQueue.reallocate_actions(
        proposed_ipfs, existing_ipfs, proposed_allocations, existing_allocs
    )
    open_allocations, open_ipfs = ActionQueue.allocate_actions(
        proposed_ipfs, reallocate_ipfs, proposed_allocations
    )
    close_allocations, close_ipfs = ActionQueue.unallocate_actions(
        existing_allocs, existing_ipfs, reallocate_ipfs, frozenlist
    )
    actions = vcat(reallocations, open_allocations, close_allocations)

    # Send ActionQueue inputs to indexer management server
    client = Client(management_server_url)
    response = mutate(client, "queueActions", Dict("actions" => actions))

    return response
end

function create_rules!(
    indexer_id::AbstractString,
    indexer_service_network_url::T,
    proposed_allocations::Dict{T,<:Real},
    whitelist::AbstractVector{T},
    blacklist::AbstractVector{T},
    pinnedlist::AbstractVector{T},
    frozenlist::AbstractVector{T},
) where {T<:AbstractString}
    actions = []

    # Query existing allocations that are not frozen
    existing_allocations = query_indexer_allocations(
        Client(indexer_service_network_url), indexer_id
    )
    existing_allocs::Dict{String,String} = Dict(
        ipfshash.(existing_allocations) .=> id.(existing_allocations)
    )
    existing_ipfs::Vector{String} = ipfshash.(existing_allocations)
    proposed_ipfs::Vector{String} = collect(keys(proposed_allocations))

    # Generate CLI commands
    reallocations, reallocate_ipfs = CLI.reallocate_actions(
        proposed_ipfs, existing_ipfs, proposed_allocations, existing_allocs
    )
    existing_ipfs ∩ proposed_ipfs
    open_allocations, open_ipfs = CLI.allocate_actions(
        proposed_ipfs, reallocate_ipfs, proposed_allocations
    )
    close_allocations, close_ipfs = CLI.unallocate_actions(
        existing_ipfs, reallocate_ipfs, frozenlist
    )
    actions = vcat(reallocations, open_allocations, close_allocations)
    return actions
end

end
