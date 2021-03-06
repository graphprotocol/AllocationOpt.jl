using AllocationOpt
using Test
using GraphQLClient

@testset "AllocationOpt.jl" begin
    include("../src/exceptions.jl")
    include("../src/domainmodel.jl")
    include("../src/query.jl")
    include("../src/service.jl")
    include("../src/ActionQueue.jl")

    gateway_url = "https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet"

    # Tests
    include("domainmodel.jl")
    include("query.jl")
    include("service.jl")
    include("actionqueue.jl")

    @testset "network_state" begin
        client = Client(gateway_url)
        id = query(client, "indexers"; query_args=Dict("where" => Dict("stakedTokens_gte" => "100000000000000000000000")), output_fields="id").data["indexers"][1]["id"]

        @test_throws AllocationOpt.UnknownIndexerError network_state(
            "0x6ac85b9d834b51b14a7b0ed849bb5199e",
            1,
            String[],
            String[],
            String[],
            String[],
            gateway_url,
        )
        @test_throws AllocationOpt.BadSubgraphIpfsHashError network_state(
            id, 1, String[""], String[], String[], String[], gateway_url
        )
    end

    @testset "optimize_indexer" begin
        filter_fn = (a, b, c) -> a[end, :]
        τ = 0.0
        network_id = 1
        client = Client(gateway_url)
        id = query(client, "indexers"; query_args=Dict("where" => Dict("stakedTokens_gte" => "100000000000000000000000")), output_fields="id").data["indexers"][1]["id"]
        all_hashes = [
            deployment["ipfsHash"] for deployment in
            query(client, "subgraphDeployments"; query_args=Dict("first" => 1000, "where" => Dict("signalledTokens_gte" => "100000000000000000000000")), output_fields="ipfsHash").data["subgraphDeployments"]
        ]
        ipfshash = all_hashes[1]
        another_ipfshash = all_hashes[2]

        # Indexer stake should be equal to the sum of allocations
        # query indexer's stake
        indexer = query(client, "indexers"; query_args=Dict("where" => Dict("id" => id)), output_fields=["delegatedTokens", "stakedTokens"]).data["indexers"][1]
        stake = togrt(indexer["delegatedTokens"]) + togrt(indexer["stakedTokens"])

        # run optimize_indexer
        repo, optindexer, network = network_state(
            id, network_id, String[ipfshash], String[], String[], String[], gateway_url
        )
        allocs = optimize_indexer(optindexer, repo, repo, 2, τ, filter_fn)
        # Sum allocation amounts
        ω = sum(values(allocs))
        @test isapprox(ω, stake; atol=1e-6)
        # Length of allocations is 1
        @test length(allocs) == 1
        # run optimize_indexer
        repo, optindexer, network = network_state(
            id,
            network_id,
            String[ipfshash, another_ipfshash],
            String[],
            String[],
            String[],
            gateway_url,
        )
        allocs = optimize_indexer(optindexer, repo, repo, 2, τ, filter_fn)
        # Sum allocation amounts
        ω = sum(values(allocs))
        @test isapprox(ω, stake; atol=1e-6)
        # Length of allocations ≤ length of whitelist
        @test length(allocs) ≤ 2
    end

    @testset "read_filterlists" begin
        # Should handle CSV input
        cols = read_filterlists("example.csv")
        @test length(cols[1]) == 2
    end
end
