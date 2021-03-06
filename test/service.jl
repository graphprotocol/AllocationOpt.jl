@testset "service" begin
    @testset "detach_indexer" begin
        repo = Repository(
            [
                Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)]),
                Indexer("0x01", 20.0, [Allocation("Qmaaa", 10.0, 0)]),
            ],
            [SubgraphDeployment("1x00", "Qmaaa", 10.0)],
        )

        # Should detach the specified indexer
        id = "0x00"
        indexer, frepo = detach_indexer(repo, id)
        @test frepo.indexers[1].id == "0x01"
        @test length(frepo.indexers) == 1
        @test indexer.id == "0x00"

        # Should throw an exception when `id` not in repo
        id = "0x10"
        @test_throws UnknownIndexerError detach_indexer(repo, id)
    end

    @testset "signal" begin
        subgraphs = [SubgraphDeployment("1x00", "Qmaaa", 10.0)]

        # Should get the signal of the one defined subgraph
        @test signal.(subgraphs) == [10.0]
    end

    @testset "ipfshash" begin
        subgraphs = [SubgraphDeployment("1x00", "Qmaaa", 10.0)]

        # Should get the ipfshash of the one defined subgraph
        @test ipfshash.(subgraphs) == ["Qmaaa"]
    end

    @testset "allocation" begin
        indexers = [
            Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)]),
            Indexer("0x01", 20.0, [Allocation("Qmaaa", 20.0, 0)]),
        ]

        # Should get the allocations of indexers
        @test allocation.(indexers) ==
            [[Allocation("Qmaaa", 10.0, 0)], [Allocation("Qmaaa", 20.0, 0)]]
    end

    @testset "allocated_stake" begin
        allocs = [Allocation("Qmaaa", 10.0, 0), Allocation("Qmaaa", 20.0, 0)]

        # Should get the allocations of indexers
        @test allocated_stake.(allocs) == [10.0, 20.0]
    end

    @testset "stakes" begin
        repo = Repository(
            [
                Indexer(
                    "0x00",
                    10.0,
                    [Allocation("Qmaaa", 10.0, 0), Allocation("Qmbbb", 10.0, 0)],
                ),
                Indexer("0x01", 20.0, [Allocation("Qmaaa", 10.0, 0)]),
            ],
            [SubgraphDeployment("1x00", "Qmaaa", 10.0)],
        )
        # Should get the sum of stake of the one defined subgraph
        @test stakes(repo) == [20.0]

        repo = Repository(
            [
                Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)]),
                Indexer("0x01", 20.0, [Allocation("Qmccc", 8.0, 0)]),
            ],
            [
                SubgraphDeployment("1x00", "Qmaaa", 10.0),
                SubgraphDeployment("1x01", "Qmbbb", 5.0),
                SubgraphDeployment("1x01", "Qmccc", 5.0),
            ],
        )

        # Should get proper sum of stake according to the subgraph ids
        @test stakes(repo) == [10.0, 0.0, 8.0]
    end

    @testset "optimize" begin
        repo = Repository(
            [
                Indexer(
                    "0x01", 10.0, [Allocation("Qmaaa", 1.0, 0), Allocation("Qmbbb", 7.0, 0)]
                ),
            ],
            [
                SubgraphDeployment("1x00", "Qmaaa", 10.0),
                SubgraphDeployment("1x01", "Qmbbb", 5.0),
            ],
        )
        indexer = Indexer("0x00", 5.0, Allocation[])
        ?? = stakes(repo)
        ?? = signal.(repo.subgraphs)
        ?? = stake(indexer)
        ?? = optimize(??, ??, ??)
        @test isapprox(??, [4.2, 0.8], atol=0.1)
    end

    @testset "projectsimplex" begin
        # Shouldn't project since already on simplex
        x = [5, 2, 8]
        z = 15
        @test projectsimplex(x, z) == x

        # Should set negative value to zero and scale others up
        # to be on simplex
        x = [-5, 2, 8]
        z = 15
        w = projectsimplex(x, z)
        @test sum(w) == z
        @test all(w .??? 0)
        @test w[1] < w[2] < w[3]

        # Should scale values down to be on simplex
        x = [20, 2, 8]
        z = 15
        w = projectsimplex(x, z)
        @test sum(w) == z
        @test all(w .??? 0)
        @test w[2] < w[3] < w[1]
    end

    @testset "???f" begin
        # ?? is 0
        ?? = Float64[5, 2, 8]
        ?? = Float64[2, 1, 1]
        ?? = Float64[0, 0, 0]
        df = ???f.(??, ??, ??)
        @test df ??? [-2.5, -2, -8]
    end

    @testset "discount" begin
        # ?? = 1.0
        ?? = Float64[2, 5, 3]
        ?? = Float64[7, 2, 1]
        ?? = 10.0
        ?? = 1.0
        ??new = discount(??, ??, ??, ??)
        ??0 = zeros(length(??))
        @test ??new == optimize(??0, ??, ??)

        # ?? = 0.0
        ?? = 0.0
        ??new = discount(??, ??, ??, ??)
        @test ??new == ??

        # ?? = 0.2, result should still sum to ?? because of simplex projection
        ?? = 0.2
        ??new = discount(??, ??, ??, ??)
        @test sum(??new) == ??
    end

    @testset "gssp" begin
        # Shouldn't project since already on simplex
        x = [5, 2, 8, 0, 1]
        k = 3
        ?? = 15
        @test gssp(x, k, ??) == [5, 2, 8, 0, 0]

        # Should set negative value to zero and scale others up
        # to be on simplex
        x = [-5, 2, 8, -10, -8]
        k = 3
        ?? = 15
        @test gssp(x, k, ??) == [0, 4.5, 10.5, 0, 0]

        # Should scale values down to be on simplex
        x = [20, 2, 8, 1, 7]
        k = 3
        ?? = 15
        w = gssp(x, k, ??)
        @test sum(w) ??? ??
        @test all(w .??? 0)
        @test w[1] > w[3] > w[5]
    end

    @testset "pgd_step" begin
        # ?? is 0
        ?? = Float64[5, 2, 8]
        ?? = Float64[2, 1, 1]
        ?? = Float64[0, 0, 0]
        k = 2
        ?? = 15
        ?? = 1
        ????? = pgd_step(??, ??, ??, k, ??, ??)
        @test ????? ??? [4.75, 0.0, 10.25]

        # ?? is 0
        ?? = Float64[5, 2, 8]
        ?? = Float64[2, 1, 1]
        ?? = Float64[0, 0, 0]
        k = 2
        ?? = 15
        ?? = 0
        ????? = pgd_step(??, ??, ??, k, ??, ??)
        @test ????? ??? [7.5, 7.5, 0.0]
    end

    @testset "pgd" begin
        ?? = Float64[5, 2, 8]
        ?? = Float64[2, 1, 1]
        k = 1
        ?? = 15
        ?? = 10000
        ???? = 1.001
        patience = 1e4
        tol = 1e-3
        ????? = pgd(??, ??, k, ??, ??, ????, patience, tol)
        @test ????? == [0.0, 0.0, 15.0]
    end

    @testset "optimize pgd" begin
        # k = 2
        filter_fn = (a, b, c) -> a[end, :]
        a = 1e5
        b = 1e7
        ?? = Float64[5, 8] * a
        ?? = Float64[2, 1] * b
        ?? = 15.0 * b
        ??opt = optimize(??, ??, ??)
        ?? = Float64[5, 2, 8] * a
        ?? = Float64[2, 1, 1] * b
        k = 2
        ?? = optimize(??, ??, ??, k, filter_fn)
        @test isapprox(??opt, ??[findall(?? .!= 0.0)]; rtol=1e-2)
    end
end
