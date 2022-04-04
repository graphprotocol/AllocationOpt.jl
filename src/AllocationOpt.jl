module AllocationOpt
using Distributed
export allocation_optimization

export optimize_indexer_to_csv!

using CSV
using DataFrames

include("exceptions.jl")
include("graphrepository.jl")
include("data.jl")
include("gascost.jl")
include("optimize.jl")

function optimize_indexer_to_csv!(;
    id::String,
    whitelist::Union{Nothing,Vector{String}},
    blacklist::Union{Nothing,Vector{String}},
    csv_write_path::String,
)
    url = "https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet"
    repository = snapshot(; url=url, indexer_query=nothing, subgraph_query=nothing)
    alloc, filtered = optimize(id, repository, whitelist, blacklist)
    df = DataFrame(
        "Subgraph ID" => collect(keys(alloc)), "Allocation in GRT" => collect(values(alloc))
    )
    df[!, "Subgraph Signal"] = map(x -> x.signal, filtered.subgraphs)
    return CSV.write(csv_write_path, df)
end

function filter_subgraphs(repo::Repository, allocList::Dict{String, Float64}, threshold::Float64)
  return Repository(
    repo.indexers, 
    filter(x -> x.id in keys(allocList) && allocList[x.id] > threshold, repo.subgraphs)
  )
end

function allocation_optimization(optimizeID::String, repository::Repository, gasBaseFee::Float64)
  # initial run
  allocList::Dict{String, Float64} = optimize(optimizeID, repository, nothing, nothing)[1]
  println("given: \nrepository: $(repository)\nallocList: $(allocList)\ngasBaseFee: $(gasBaseFee)")
  println("initial run - $(estimated_profit(repository, allocList, gasBaseFee))")

  # preset parameters 
  allocation_min_thresholds::Vector{Float64} = [0, gasBaseFee, gasBaseFee*3, gasBaseFee*10, gasBaseFee*50]
  println("preset parameters - $(allocation_min_thresholds)")

  # spawn jobs
  for threshold in allocation_min_thresholds
    filtered_repo = filter_subgraphs(repository, allocList, threshold)
    println("gas base fee - $(threshold)")
    if length(filtered_repo.subgraphs) > 0
      plan = @spawn optimize(
        optimizeID, 
        filtered_repo, 
        nothing, nothing)
      result = fetch(plan)[1]
      profit = estimated_profit(filtered_repo, result, gasBaseFee)
      println("result: $(result) \nprofit: $(profit) \n")

       # keep track of the most profitable allocation 
      if profit >= estimated_profit(repository, allocList, gasBaseFee)
        allocList = result
      end
    end
  end
  return allocList
end
end
