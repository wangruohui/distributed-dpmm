using JLD
using HDF5
using Clustering

@everywhere begin
    include("HyperParameter.jl")
    include("Cluster.jl")
    include("Gaussian.jl")
    include("Multinomial.jl")
    include("Probabilities.jl")
    include("mysample.jl")
    include("Worker.jl")
    include("Center.jl")
    include("ExperimentSetting.jl")
    include("Evaluation.jl")
end

@everywhere function evalCRP(M::Int, T::Int, dataset::Symbol, PRINT::Bool=false)
    # srand(10)

    ## Import Experiment Setting
    data,gt,logα,hp = get_exp_setting(dataset)

    ndata = length(gt)
    assert(size(data, 2) == length(gt))
    PRINT && println("Size of Data: ", size(data))
    PRINT && println("Parameter, logα = ", logα)
    # PRINT && println("Parameter, hp = ", hp)

    ## Init worker
    wk = Worker(data,1,1,hp,rand(1:M,ndata))

    ## Evaluation
    like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
    lll = Array{Float64}(T+1)
    vi = Array{Float64}(T+1)
    runtime = Array{Float64}(T)
    count = Array{Float64}(T+1)

    ## T iteration of CRP
    for iter = 1:T+1
        lll[iter] = like_data + likelihood_cluster(collect(values(wk.dcc)), logα, hp)
        vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(wk.label),wk.label)
        count[iter] = length(wk.dcc)

        PRINT && println()
        PRINT && println(" loglikelihood = ", lll[iter])
        PRINT && println(" vi = ", vi[iter])
        PRINT && println(" count = ", count[iter])
        PRINT && println("Start CRP")

    iter == T+1 && break

    PRINT && println()
    PRINT && println("Iter = ", iter)

        tic()
        crp!(wk,hp,logα)
        runtime[iter] = toq()
        PRINT && println(" runtime = ", runtime[iter])

        PRINT && report_cluster_size(wk)
    end

    PRINT && println(vi)
    PRINT && println(lll)

    return lll, vi, runtime, count
end

function evalCRPrepeat(R::Int, T::Int, dataset::Symbol, print::Bool=false)
    assert(R<nworkers())

    lll = Array{Float64}(T+1,R)
    vi = Array{Float64}(T+1,R)
    runtime = zeros(Float64, T+1, R)
    count = Array{Float64}(T+1,R)

    @sync for r in 1:R
        @async lll[:,r], vi[:,r], runtime[2:end,r], count[:,r] = @fetch evalCRP(1,T,dataset,print)
    end

    lll_mean = mean(lll,2)
    vi_mean = mean(vi,2)
    runtime_mean = mean(runtime,2)
    count_mean = mean(count,2)
    cum_runtime_mean = cumsum(runtime_mean)

    lll_std = std(lll,2)
    vi_std = std(vi,2)
    runtime_std = std(runtime,2)
    count_std = std(count,2)

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
end
