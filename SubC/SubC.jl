using Distributions
using Clustering
using HDF5
using JLD

begin
    include("../HyperParameter.jl")
    include("../Cluster.jl")
    include("../Gaussian.jl")
    include("../Multinomial.jl")
    include("../mysample.jl")
    include("../Evaluation.jl")
    include("../ExperimentSetting.jl")
    include("probability.jl")
    include("utils.jl")
end

function evalSubc(T::Int, dataset::Symbol)
    # srand(10)

    ## Import Experiment Setting
    data,gt,logα,hp = get_exp_setting(dataset)
    ndata = length(gt)

    ## Evaluation
    like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
    lll = Array{Float64}(T+1)
    vi = Array{Float64}(T+1)
    count = Array{Int}(T+1)
    runtime = Array{Float64}(T)
    bytes_transfer = zeros(Int,T)
    times_transfer = zeros(Int,T)

    ## Init state
    init_logπ = rand(2, 3)
    init_θ = rand(hp.ds, 2, 3)
    for i in 1:size(init_θ,2), j in 1:size(init_θ,3)
        init_θ[:,i,j] ./= sum(init_θ[:,i,j])
    end
    z, N, ss = sample_assignment(data, init_logπ, init_θ, logα, hp)

    ## T iteration of CRP
    for iter = 1:T+1
        cc = Vector{Cluster}(0)
        for k in 1:size(N,1)
            push!(cc, Cluster(N[k,1], ss[:,k,1]))
        end

        lll[iter] = like_data + likelihood_cluster(cc, logα, hp)
        vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(z[:,1]),z[:,1])
        count[iter] = size(N,1)

        println()
        println(" loglikelihood = ", lll[iter])
        println(" vi = ", vi[iter])
        println(" count = ", count[iter])

        iter == T+1 && break

        println()
        println("Iter = ", iter)

        tic()

        @time logπ,θ = sample_parameter(N, ss, logα, hp)
        @time z, N, ss = sample_assignment(data, logπ, θ, logα, hp)

        assert(sum(N[:,1]) == ndata)
        assert(sum(ss[:,:,1]) - sum(data) < eps(Float32))
        assert(N[:,1] == N[:,2] + N[:,3])
        assert(sum(abs2, ss[:,:,1]-ss[:,:,2]-ss[:,:,3]) < eps(Float32))

        N, ss = split_subcluster(data, z, N, ss, logα, hp)
        N, ss = remove_empty(N, ss)

        runtime[iter] = toq()

        println(sort(N[:,1]))
    end

    println()
    println("vi = ", vi)
    println("loglikelihood = ", lll)
    println("runtime = ", runtime)
    println("bytes_transfer = ", bytes_transfer)

    return lll, vi, runtime, count, bytes_transfer
end


function evalSubcRepeat(R::Int, T::Int, dataset::Symbol)
    assert(R <= nworkers())

    lll = Array{Float64}(T, R+1)
    vi = Array{Float64}(T, R+1)
    runtime = zeros(Float64, T, R)
    count = Array{Float64}(T, R+1)
    bytes_transfer = Array{Float64}(T, R)

    @sync for r in 1:R
        println("Repeat = ", r)
        @async lll[:,r], vi[:,r], runtime[:,r], count[:,r], bytes_transfer[:,r] = @fetch evalSubc(T,dataset)
    end

    lll_mean = mean(lll,2)
    vi_mean = mean(vi,2)
    runtime_mean = mean(runtime,2)
    count_mean = mean(count,2)
    cum_runtime_mean = cumsum(vcat(0,runtime_mean))
    bytes_transfer_mean = mean(bytes_transfer,2)

    lll_std = std(lll,2)
    vi_std = std(vi,2)
    runtime_std = std(runtime,2)
    count_std = std(count,2)
    bytes_transfer_std = std(bytes_transfer,2)

    println()
    println("lll_mean = ", lll_mean)
    println("lll_std = ", lll_std)
    println("vi_mean = ", vi_mean)
    println("vi_std = ", vi_std)
    println("runtime_mean = ", runtime_mean)
    println("runtime_std = ", runtime_std)
    println("cum_runtime_mean = ", cum_runtime_mean)
    println("count_mean = ", count_mean)
    println("count_std = ", count_std)
    println("bytes_transfer_mean = ", bytes_transfer_mean)
    println("bytes_transfer_std = ", bytes_transfer_std)
end
