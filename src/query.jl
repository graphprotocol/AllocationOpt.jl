using GraphQLClient

function verify_ipfshashes(xs::AbstractVector{<:AbstractString})
    return all(verify_ipfshash.(xs))
end

function ipfshash_in(
    whitelist::AbstractVector{T}, pinnedlist::AbstractVector{T}
) where {T<:AbstractString}
    return whitelist ∪ pinnedlist
end

function ipfshash_not_in(
    blacklist::AbstractVector{T}, frozenlist::AbstractVector{T}
) where {T<:AbstractString}
    return blacklist ∪ frozenlist
end

function frozen_stake(
    client::Client, id::AbstractString, frozenlist::AbstractVector{<:AbstractString}
)
    allocations_where_query = Dict("status" => "Active", "indexer" => id)

    allocations_query = GQLQuery(
        Dict("where" => allocations_where_query),
        ["allocatedTokens", "subgraphDeployment{ ipfsHash }"],
    )
    allocations_data = query(client, "allocations"; query_args=allocations_query.args, output_fields=allocations_query.fields).data["allocations"]
    return sum(
        map(
            a -> togrt(a["allocatedTokens"]),
            filter(
                a -> (a["subgraphDeployment"]["ipfsHash"] in frozenlist), allocations_data
            ),
        ),
    )
end

function query_subgraphs(
    client::Client, ipfsin::AbstractVector{T}, ipfs_notin::AbstractVector{T}
) where {T<:AbstractString}
    subgraph_where_query::Dict{AbstractString,Union{AbstractString,AbstractVector{<:AbstractString}}} = Dict(
        "signalledTokens_gte" => "1000000000000000000000"
    )
    if !isempty(ipfsin)
        subgraph_where_query["ipfsHash_in"] = ipfsin
    end
    if !isempty(ipfs_notin)
        subgraph_where_query["ipfsHash_not_in"] = ipfs_notin
    end
    subgraph_query = GQLQuery(
        Dict(
            "first" => 1000,
            "orderBy" => "signalledTokens",
            "orderDirection" => "desc",
            "where" => subgraph_where_query,
        ),
        ["id", "ipfsHash", "signalledTokens"],
    )
    subgraphs_data = query(client, "subgraphDeployments"; query_args=subgraph_query.args, output_fields=subgraph_query.fields).data["subgraphDeployments"]
    subgraphs = map(
        x -> SubgraphDeployment(x["id"], x["ipfsHash"], x["signalledTokens"]),
        subgraphs_data,
    )
    return subgraphs
end

function query_indexer_allocations(client::Client, indexer_id::AbstractString)
    indexer_query = GQLQuery(
        Dict(
            "first" => 1000,
            "where" =>
                Dict("stakedTokens_gte" => "100000000000000000000000", "id" => indexer_id),
        ),
        ["allocations{id,subgraphDeployment{ipfsHash}}"],
    )
    indexer_data = query(client, "indexers"; query_args=indexer_query.args, output_fields=indexer_query.fields).data["indexers"]
    indexer = Indexer(indexer_data[1]["allocations"])
    return indexer.allocations
end

function query_indexers(client::Client, subgraphs::AbstractVector{SubgraphDeployment})
    subgraph_ids = id.(subgraphs)
    allocations_where_query = Dict(
        "status" => "Active", "subgraphDeployment_in" => subgraph_ids
    )
    allocations_query = GraphQLClient.directly_write_query_args(
        Dict("where" => allocations_where_query)
    )
    indexer_query = GQLQuery(
        Dict(
            "first" => 1000,
            "where" => Dict("stakedTokens_gte" => "100000000000000000000000"),
        ),
        [
            "id",
            "delegatedTokens",
            "stakedTokens",
            "indexingRewardCut",
            "allocations($allocations_query){allocatedTokens,createdAtEpoch,subgraphDeployment{ipfsHash}}",
        ],
    )
    indexers_data = query(client, "indexers"; query_args=indexer_query.args, output_fields=indexer_query.fields).data["indexers"]
    indexers = map(
        x -> Indexer(x["id"], x["delegatedTokens"], x["stakedTokens"], x["allocations"], x["indexingRewardCut"]),
        indexers_data,
    )
    return indexers
end

function networkparameters(client:: Client, network_id::Integer)
    network_query = GQLQuery(
        Dict("id" => network_id),
        [
            "id",
            "totalSupply",
            "networkGRTIssuance",
            "epochLength",
            "totalTokensSignalled",
            "currentEpoch",
        ],
    )
    network_data = query(client, "graphNetwork"; query_args=network_query.args, output_fields=network_query.fields).data["graphNetwork"]
    network = GraphNetworkParameters(
        network_data["id"],
        network_data["totalSupply"],
        network_data["networkGRTIssuance"],
        network_data["epochLength"],
        network_data["totalTokensSignalled"],
        network_data["currentEpoch"],
    )
    return network
end

function snapshot(
    client::Client, ipfsin::AbstractVector{T}, ipfs_notin::AbstractVector{T}
) where {T<:AbstractString}
    subgraphs = query_subgraphs(client, ipfsin, ipfs_notin)
    indexers = query_indexers(client, subgraphs)

    # Make repository
    repository = Repository(indexers, subgraphs)
    return repository
end
