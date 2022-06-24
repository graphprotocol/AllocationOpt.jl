using Roots
using LinearAlgebra

struct Patience
    maxval::Integer
    val::Integer
end

function detach_indexer(repo::Repository, id::AbstractString)::Tuple{Indexer,Repository}
    # Get requested indexer
    i = findfirst(x -> x.id == id, repo.indexers)
    if isnothing(i)
        throw(UnknownIndexerError())
    end
    indexer = repo.indexers[i]

    # Remove indexer from repository
    indexers = filter(x -> x.id != id, repo.indexers)
    frepo = Repository(indexers, repo.subgraphs)
    return indexer, frepo
end

function stakes(r::Repository)
    # Loop over subgraphs, getting their names
    subgraphs = ipfshash.(r.subgraphs)

    # Match name to indexer allocations
    ω = reduce(vcat, allocation.(r.indexers))
    Ω = Float64[]
    for subgraph in subgraphs
        subgraph_allocations = filter(x -> x.ipfshash == subgraph, ω)
        subgraph_amounts = allocated_stake.(subgraph_allocations)

        # If no match, then set to 0
        if isempty(subgraph_amounts)
            subgraph_amounts = [0.0]
        end
        stake_on_subgraph = sum(subgraph_amounts)
        append!(Ω, stake_on_subgraph)
    end
    return Ω
end

function solve_dual(Ω, ψ, σ)
    lower_bound = eps(Float64)
    upper_bound = (sum(.√(ψ .* Ω)))^2 / σ
    sol = find_zero(
        x -> sum(max.(0.0, .√(ψ .* Ω / x) .- Ω)) - σ,
        (lower_bound, upper_bound),
        Roots.Brent(),
    )
    return sol
end

solve_primal(Ω, ψ, v) = max.(0.0, .√(ψ .* Ω / v) - Ω)

function optimize(indexer::Indexer, repo::Repository)
    ψ = signal.(repo.subgraphs)
    Ω = stakes(repo)
    σ = indexer.stake

    # Solve the dual and use that value to solve the primal
    v = solve_dual(Ω, ψ, σ)
    ω = solve_primal(Ω, ψ, v)

    return ω
end

# Performs Bregman iteration
function optimize(
    indexer::Indexer,
    repo::Repository,
    ωopt::Vector{T},
    max_allocations::Integer,
    min_allocation_amount::Float64,
    network::GraphNetworkParameters,
    gas::T,
    allocation_lifetime::Integer,
    pinned_indices::Vector{<:Integer},
) where T <: Real
    # Parameters
    μ = 10000.0  # Trades off between objective and sparsity
    η_μ = 0.1  # How much to increase or decrease mu by
    max_μ_patience = 20  # How long μ will wait before changing
    μ_patience = Patience(max_μ_patience, max_μ_patience)
    max_bregman_patience = 100000  # How long Bregman will run if ω is not changing
    bregman_patience = Patience(max_bregman_patience, max_bregman_patience)
    η_α_up = 0.01  # How much to increase α by proportionally
    η_α_down = 0.5  # How much to decrease α by proportionally
    low = -1  # Lower bound of space onto which to project p
    high = 1  # Higher bound of space onto which to project p
    reserve_amount_per_subgraph = min_allocation_amount

    # Problem constants
    reserve_pinned = reserve_amount_per_subgraph * length(pinned_indices)
    σ = indexer.stake - reserve_pinned
    ψ = signal.(repo.subgraphs)
    Ω = stakes(repo)
    Ω[pinned_indices] = Ω[pinned_indices] .+ reserve_amount_per_subgraph

    ωs = bregmaniteration(ωopt, ψ, Ω, σ, μ, η_μ, μ_patience, η_α_up, η_α_down, low, high, bregman_patience, max_allocations, gas, network, allocation_lifetime)

    # Pick the ω with highest profit while satisfying min_allocation_amount
    # min_allocation_ωs = filter(ω -> minimum(nonzero(ω)) >= min_allocation_amount, ωs)
    # profits = map(x -> profit(network, gas, allocation_lifetime, x, ψ, Ω), min_allocation_ωs)
    # ω = min_allocation_ωs[findfirst(x -> x == maximum(profits), profits)]
    ω = ωs[end]
    
    # Add reserved stake to the pinned subgraphs
    ω[pinned_indices] = ω[pinned_indices] .+ reserve_amount_per_subgraph
    return ω
end

function bregmaniteration(
    ωopt::Vector{T},
    ψ::Vector{T},
    Ω::Vector{T},
    σ::T,
    μ::T,
    η_μ::T,
    μ_patience::Patience,
    η_α_up::T,
    η_α_down::T,
    low::Integer,
    high::Integer,
    bregman_patience::Patience,
    max_allocations::Integer,
    gas::T,
    network::GraphNetworkParameters,
    allocation_lifetime::Integer
) where T <: Real
    # Sanity checks for stopping conditions
    num_subgraphs = length(ψ)
    max_allocations = max_allocations > num_subgraphs ? num_subgraphs : max_allocations

    # Initialise with zeros
    p = zeros(num_subgraphs)
    p[argmax(ωopt)] = 1.0
    ω = zeros(num_subgraphs)
    ω[argmax(ωopt)] = σ
    ωbest = ω
    prev_profit = 0.0
    ωs = [ω]
    # Bregman loop
    while true
        @show length(nonzero(ω))
        @show μ
        z = subproblem(ω, μ, σ, ψ, Ω, p, η_α_up, η_α_down)
        ωtmp = projectsimplex(z, σ)
        # TODO: Find a better way to get unique nonzero els of ωtmp

        # ωixs = ω .> 0.0
        # ωtmpixs = ωtmp .> 0.0
        # unique_ωtmpixs = findall(ωtmpixs - ωixs .> 0.0)
        # unique_ωtmp = ωtmp[unique_ωtmpixs]
        
        # if isempty(unique_ωtmp)
        #     ωnew = ωtmp
        # else
        #     maxindex = findfirst(x -> x == maximum(unique_ωtmp), ωtmp)
        #     # newmax = sum(unique_ωtmp)
        #     # ωnew = zeros(length(ω))
        #     # ωnew[ωixs] = ωtmp[ωixs]
        #     ωmax = deepcopy(ω)
        #     ωmax[maxindex] = ωtmp[maxindex]
        #     ωnew = projectsimplex(ωmax, σ)
        # end
        ωnew = ωtmp

        # Update the patience and μ if no change was made
        μ_patience, μ, should_update = update_μ(
            μ_patience, μ, ω, ωnew, η_μ
        )

        if should_update
            # Update bregman patience
            if ω ≈ ωnew
                bregman_patience = Patience(
                    bregman_patience.maxval, bregman_patience.val - 1
                )
            end

            # Stopping conditions

            # Check if the new profit is decreased from the previous
            # # Set subgraphs less than min allocation to 0 and project onto simplex
            # ωgreaterixs = findall(x -> x >= min_allocation_amount, ω)
            # ωgreater = projectsimplex(ω[ωgreaterixs], σ)
            # ωprofit = zeros(length(ω))
            # ωprofit[ωgreaterixs] = ωgreater
            if length(nonzero(ω)) < length(nonzero(ωnew))
                # if profit_condition(prev_profit, curr_profit)
                #     # Only break on this condition when we try to add a new subgraph as
                #     # Bregman can initially allocate below the min threshold and then
                #     # increase it over time.
                #     println("Stopped due to decrease in profit")
                #     ω = ωbest
                # end
                # ωbest = ω
                # prev_profit = curr_profit
                # μ = 100000.0
                push!(ωs, ω)
            end

            # Check if tried to allocate more than max allocations or if our bregman patience has run out
            # if max_allocations_condition(ωnew, max_allocations) || bregman_patience.val == 0
            if bregman_patience.val == 0
                println("Stopped due to allocating more than maximum allocating or out of patience")
                break
            end
            
            # Update ω, p and prev_profit
            p_update = (-1 / μ) .* ∇f.(ω, ψ, Ω, μ, p)
            p = projectrange.(low, high, p_update)
            ω = ωnew
        end
    end
    return ωs
end

function subproblem(ω, μ, σ, ψ, Ω, p, η_α_up, η_α_down)
    α = 1.0
    z = ω
    zprev = fill(typemax(ω[1]), length(ω))
    while true
        znew = subproblemiteration(z, μ, σ, ψ, Ω, p, α)

        if znew ≈ z
            break
        end
        
        # Update z and α
        rprev = LinearAlgebra.norm(z - zprev)
        rcurr = LinearAlgebra.norm(znew - z)
        if rcurr ≤ rprev
            zprev = z
            z = znew
            α = α * (1.0 + η_α_up)
        else
            α = α * (1.0 - η_α_down)
        end
    end
    return z
end

function subproblemiteration(z, μ, σ, ψ, Ω, p, α)
    ξ = projectsimplex(z, σ)
    λ = compute_λ(ξ, ψ, Ω)
    y = shrink.(2 .* ξ .- z .- λ .* ∇f.(ξ, ψ, Ω, μ, p), μ .* λ)
    return z + α .* (y - ξ)
end

shrink(z::T, α) where {T<:Real} = sign(z) .* max(abs(z) - α, zero(T))

compute_λ(ω, ψ, Ω) = max(minimum(nonzero(((ω .+ Ω) .^ 3) ./ (2 .* ψ))), 1)

∇f(ω::T, ψ, Ω, μ, p) where {T<:Real} = -((ψ * Ω) / (ω + Ω + eps(T))^2) - (μ * p)

projectrange(low, high, x::T) where {T<:Real} = max(min(x, one(T) * high), one(T) * low)

# Projection onto σ-simplex
# Reference: Modified from https://gist.github.com/lendle/8564850
function projectsimplex(x::Vector{T}, σ) where {T<:Real}
    n = length(x)
    ζ = sort(x; rev=true)
    #finding ρ could be improved to avoid so much temp memory allocation
    ρ = maximum((1:n)[ζ - (cumsum(ζ) .- σ) ./ (1:n) .> zero(T)])
    λ = (sum(ζ[1:ρ]) - σ) / ρ
    z = max.(x .- λ, zero(T))
    return z
end

# This condition triggers on greater than so that Bregman can converge before it exits
max_allocations_condition(ω, max_allocations) = length(nonzero(ω)) > max_allocations

# This condition triggers on when profit starts to decrease before it exits
profit_condition(prev_profit::T, curr_profit::T) where T <: Real = prev_profit > curr_profit + 1e-6#+ eps(T)

nonzero(v::Vector{<:Real}) = v[findall(v .!= 0.0)]

function update_μ(patience, μ, x, xnew, update_rate)
    should_update = true
    # If x isn't changing, decrease μ
    # This operation is patient as continuing bregman may add a new subgraph.
    if x ≈ xnew
        patience = Patience(patience.maxval, patience.val - 1)
        if patience.val == 0
            μ = μ * (1.0 - update_rate)
            patience = Patience(patience.maxval, patience.maxval)
        end
    end

    # If we added two subgraphs, increase μ. This operation is impatient
    # as continuing bregman won't take a subgraph away.
    # if length(nonzero(xnew)) ≥ length(nonzero(x)) + 2
    #     # should_update = false
    #     μ = μ * (1.0 + update_rate)
    #     patience = Patience(patience.maxval, patience.maxval)
    # end
    
    return patience, μ, should_update
end

function tokens_issued_over_lifetime(network::GraphNetworkParameters, allocation_lifetime::Integer)
    return network.principle_supply *
           network.issuance_rate_per_block^(
        network.block_per_epoch * allocation_lifetime
    ) - network.principle_supply
end

function profit(
    network::GraphNetworkParameters,
    gas::Float64,
    allocation_lifetime::Integer,
    ω::Vector{T},
    ψ::Vector{T},
    Ω::Vector{T},
) where T <: Real
    Φ = tokens_issued_over_lifetime(network, allocation_lifetime)
    gascost = gaspersubgraph(gas) * length(nonzero(ω))
    indexing_rewards = f(ψ, Ω, ω, Φ, network.total_tokens_signalled)
    # @show gascost
    # @show indexing_rewards
    return indexing_rewards - gascost
end

function gaspersubgraph(gas)
    # As of now, assume cost for an allocation's life require open, close, and claim
    # where claim is the 0.3 times open or close
    open_multiplier = 1.0
    close_multiplier = 1.0
    claim_multiplier = 0.3
    return open_multiplier * gas + close_multiplier * gas + claim_multiplier * gas
end

function f(repo::Repository, ω)
    ψ = signal.(repo.subgraphs)
    Ω = stakes(repo)

    return f(ψ, Ω, ω)
end

function f(ψ::Vector{T}, Ω::Vector{T}, ω::Vector{T}, Φ::T, Ψ::T) where T <: Real
    subgraph_rewards = Φ .* ψ ./ Ψ
    indexing_rewards = sum(subgraph_rewards .* ω ./ (Ω .+ ω))
    return indexing_rewards
end

function f(ψ::Vector{T}, Ω::Vector{T}, ω::Vector{T}) where {T<:Real}
    return sum((ψ .* ω) ./ (ω .+ Ω .+ eps(T)))
end
