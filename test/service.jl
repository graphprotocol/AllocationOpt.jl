@testset "service" begin
    # @testset "detach_indexer" begin
    #     repo = Repository(
    #         [
    #             Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)], 0.0),
    #             Indexer("0x01", 20.0, [Allocation("Qmaaa", 10.0, 0)], 0.0),
    #         ],
    #         [SubgraphDeployment("1x00", "Qmaaa", 10.0)],
    #     )

    #     # Should detach the specified indexer
    #     id = "0x00"
    #     indexer, frepo = detach_indexer(repo, id)
    #     @test frepo.indexers[1].id == "0x01"
    #     @test length(frepo.indexers) == 1
    #     @test indexer.id == "0x00"

    #     # Should throw an exception when `id` not in repo
    #     id = "0x10"
    #     @test_throws UnknownIndexerError detach_indexer(repo, id)
    # end

    # @testset "signal" begin
    #     subgraphs = [SubgraphDeployment("1x00", "Qmaaa", 10.0)]

    #     # Should get the signal of the one defined subgraph
    #     @test signal.(subgraphs) == [10.0]
    # end

    # @testset "ipfshash" begin
    #     subgraphs = [SubgraphDeployment("1x00", "Qmaaa", 10.0)]

    #     # Should get the ipfshash of the one defined subgraph
    #     @test ipfshash.(subgraphs) == ["Qmaaa"]
    # end

    # @testset "allocation" begin
    #     indexers = [
    #         Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)], 0.0),
    #         Indexer("0x01", 20.0, [Allocation("Qmaaa", 20.0, 0)], 0.0),
    #     ]

    #     # Should get the allocations of indexers
    #     @test allocation.(indexers) ==
    #         [[Allocation("Qmaaa", 10.0, 0)], [Allocation("Qmaaa", 20.0, 0)]]
    # end

    # @testset "allocated_stake" begin
    #     allocs = [Allocation("Qmaaa", 10.0, 0), Allocation("Qmaaa", 20.0, 0)]

    #     # Should get the allocations of indexers
    #     @test allocated_stake.(allocs) == [10.0, 20.0]
    # end

    # @testset "stakes" begin
    #     repo = Repository(
    #         [
    #             Indexer(
    #                 "0x00",
    #                 10.0,
    #                 [Allocation("Qmaaa", 10.0, 0), Allocation("Qmbbb", 10.0, 0)],
    #                 0.0,
    #             ),
    #             Indexer("0x01", 20.0, [Allocation("Qmaaa", 10.0, 0)], 0.0),
    #         ],
    #         [SubgraphDeployment("1x00", "Qmaaa", 10.0)],
    #     )
    #     # Should get the sum of stake of the one defined subgraph
    #     @test stakes(repo) == [20.0]

    #     repo = Repository(
    #         [
    #             Indexer("0x00", 10.0, [Allocation("Qmaaa", 10.0, 0)], 0.0),
    #             Indexer("0x01", 20.0, [Allocation("Qmccc", 8.0, 0)], 0.0),
    #         ],
    #         [
    #             SubgraphDeployment("1x00", "Qmaaa", 10.0),
    #             SubgraphDeployment("1x01", "Qmbbb", 5.0),
    #             SubgraphDeployment("1x01", "Qmccc", 5.0),
    #         ],
    #     )

    #     # Should get proper sum of stake according to the subgraph ids
    #     @test stakes(repo) == [10.0, 0.0, 8.0]
    # end

    # @testset "optimize naive" begin
    #     repo = Repository(
    #         [
    #             Indexer(
    #                 "0x01", 10.0, [Allocation("Qmaaa", 2.0, 0), Allocation("Qmbbb", 8.0, 0)], 0.0
    #             ),
    #         ],
    #         [
    #             SubgraphDeployment("1x00", "Qmaaa", 10.0),
    #             SubgraphDeployment("1x01", "Qmbbb", 5.0),
    #         ],
    #     )
    #     indexer = Indexer("0x00", 5.0, Allocation[], 0.0)
    #     ω = optimize(indexer, repo)
    #     @test isapprox(ω, [4.2, 0.8], atol=0.1)
    # end

    # @testset "projectsimplex" begin
    #     # x is already on the simplex
    #     x = [5, 2, 8]
    #     σ = 15
    #     z = projectsimplex(x, σ)
    #     @test z == x

    #     # x needs to be projected
    #     x = [-5, 2, 8]
    #     σ = 15
    #     z = projectsimplex(x, σ)
    #     @test z == [0, 4.5, 10.5]
    # end

    # @testset "projectrange" begin
    #     # x has elements that must be projected
    #     x = [-5, 2, 0.8]
    #     low = -1
    #     high = 1
    #     z = projectrange.(low, high, x)
    #     @test z == [-1, 1, 0.8]
    # end

    # @testset "shrink" begin
    #     # Case z all positive
    #     z = [5, 2, 8]
    #     α = 0
    #     y = shrink.(z, α)
    #     @test y == z

    #     # Case z contains negatives
    #     z = [-5, -2, -8]
    #     α = 0
    #     y = shrink.(z, α)
    #     @test y == z

    #     # Case α pulls max below 0
    #     z = [5, 2, 8]
    #     α = 3
    #     y = shrink.(z, α)
    #     @test y == [2, 0, 5]
    # end

    # @testset "min_allocation_condition" begin
    #     # Allocations above min
    #     ω = [5, 2, 8, 0]
    #     minimum_allocation_amount = 0
    #     condition = min_allocation_condition(ω, minimum_allocation_amount)
    #     @test !condition

    #     # Allocations below min
    #     ω = [5, 2, 8, 0]
    #     minimum_allocation_amount = 3
    #     condition = min_allocation_condition(ω, minimum_allocation_amount)
    #     @test condition
    # end

    # @testset "max_allocations_condition" begin
    #     # Max 5 allocations, so condition not met
    #     ω = [5, 2, 8]
    #     max_allocations = 5
    #     condition = max_allocations_condition(ω, max_allocations)
    #     @test !condition

    #     # Max 2 allocations, so condition met
    #     ω = [5, 2, 8]
    #     max_allocations = 2
    #     condition = max_allocations_condition(ω, max_allocations)
    #     @test condition

    #     # Max 3 allocations, but this condition only triggers on greater than so condition not met
    #     ω = [5, 2, 8]
    #     max_allocations = 3
    #     condition = max_allocations_condition(ω, max_allocations)
    #     @test !condition
    # end

    # @testset "compute_λ" begin
    #     ψ = [5, 2, 8, 1]
    #     Ω = [2, 1, 1, 0]
    #     ω = [0, 0, 0, 0]
    #     λ = compute_λ(ω, ψ, Ω)
    #     @test λ == 1 / 16
    # end

    # @testset "∇f" begin
    #     # ω and p are 0
    #     ψ = Float64[5, 2, 8]
    #     Ω = Float64[2, 1, 1]
    #     ω = Float64[0, 0, 0]
    #     p = Float64[0, 0, 0]
    #     μ = 0.1
    #     df = ∇f.(ω, ψ, Ω, μ, p)
    #     @test df ≈ [-2.5, -2, -8]

    #     # ω and p are 1
    #     ψ = Float64[5, 2, 8]
    #     Ω = Float64[2, 1, 1]
    #     ω = Float64[1, 1, 1]
    #     p = Float64[1, 1, 1]
    #     μ = 0.1
    #     df = ∇f.(ω, ψ, Ω, μ, p)
    #     @test df == [-10 / 9 - 0.1, -0.6, -2.1]
    # end

    # @testset "subproblemiteration" begin
    #     # z and p are 0
    #     ψ = Float64[5, 2, 8]
    #     Ω = Float64[2, 1, 1]
    #     z = Float64[10, 0, 0]
    #     p = Float64[1.0, 0, 0]
    #     μ = 100.0
    #     σ = 10.0
    #     # λ = 1/16
    #     # z_new = subproblemiteration(z, μ, σ, ψ, Ω, p, λ)
    #     z_new = subproblemiteration(z, μ, σ, ψ, Ω, p)
    #     @test z_new ≈ [11525 / 1152, 0, 0]
    # end

    # @testset "update μ" begin
    #     # Patience should decrease
    #     patience = Patience(10, 10)
    #     μ = 10.0
    #     update_rate = 0.1
    #     xnew = [1, 2, 3]
    #     x = [1, 2, 3]
    #     patience, μ, should_update = update_μ(patience, μ, x, xnew, update_rate)
    #     @test patience.val == 9
    #     @test should_update

    #     # Patience should reset and μ should decrease by 10%
    #     patience = Patience(10, 1)
    #     μ = 10.0
    #     update_rate = 0.1
    #     xnew = [1, 2, 3]
    #     x = [1, 2, 3]
    #     patience, μ, should_update = update_μ(patience, μ, x, xnew, update_rate)
    #     @test patience.val == 10
    #     @test μ == 9.0
    #     @test should_update

    #     # Patience should reset and μ should increase by 10%
    #     patience = Patience(10, 1)
    #     μ = 10.0
    #     update_rate = 0.1
    #     xnew = [1, 2, 3]
    #     x = [1, 0, 0]
    #     patience, μ, should_update = update_μ(patience, μ, x, xnew, update_rate)
    #     @test patience.val == 10
    #     @test μ == 11.0
    #     @test !should_update
    # end

    @testset "optimize bregman" begin
        repo = Repository(
            [
                Indexer(
                    "0x01",
                    10.0,
                    [
                        Allocation("Qmaaa", 2.0, 0),
                        Allocation("Qmbbb", 8.0, 0),
                        Allocation("Qmccc", 1.0, 0),
                    ],
                    0.0,
                ),
            ],
            [
                SubgraphDeployment("1x00", "Qmaaa", 2.0),
                SubgraphDeployment("1x01", "Qmbbb", 5.0),
                SubgraphDeployment("1x02", "Qmccc", 8.0),
            ],
        )
        indexer = Indexer("0x00", 5.0, Allocation[], 0.0)
        ωopt = optimize(indexer, repo)
        network = GraphNetworkParameters("1", 100.0, 1.0001, 30, 15.0, 0)

        # Should match the analytic solution
        max_allocations = 100
        min_allocation_amount = -1.0
        gas = 0.0
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test isapprox(ω, ωopt, atol=0.1)

        # Should stop at two allocations
        max_allocations = 2
        min_allocation_amount = -1.0
        gas = 0.0
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test length(nonzero(ω)) == 2

        # Should stop at one allocation
        max_allocations = 1
        min_allocation_amount = -1.0
        gas = 0.0
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test length(nonzero(ω)) == 1

        # Should stop when it tries to allocate less than 1.0
        max_allocations = 100
        min_allocation_amount = 1.0
        ωopt = optimize(indexer, repo)
        gas = 0.0
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test minimum(nonzero(ω)) > min_allocation_amount

        # Should stop at two subgraphs since profit decreases
        max_allocations = 2
        min_allocation_amount = -1.0
        gas = 0.001
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test length(nonzero(ω)) == 2

        # Should stop at one subgraph since profit decreases 
        max_allocations = 2
        min_allocation_amount = 0.0
        gas = 0.1
        allocation_lifetime = 1
        pinned_ixs = Int64[]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test length(nonzero(ω)) == 1

        # Should stop at two allocations but add one back
        max_allocations = 2
        min_allocation_amount = -1.0
        gas = 0.0
        allocation_lifetime = 1
        pinned_ixs = Int64[2]
        ω = optimize(indexer, repo, ωopt, max_allocations, min_allocation_amount, network, gas, allocation_lifetime, pinned_ixs)
        @test length(nonzero(ω)) == 3
    end
end
