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
    BLAS.set_num_threads(1)
end

using JLD
using HDF5
using Distributions
using Clustering

function evalAsyncDistribute(M::Int, T::Int, TT::Int, dataset::Symbol, eval::Bool=true)
    assert(M<=nworkers())

    ## Import Experiment Setting
    data,gt,logα,hp = get_exp_setting(dataset)
    ndata = length(gt)

    ## Init workers
    indexrange = Array{Range{Int}}(M)
    fwks = Array{Future}(M)
    @sync for i in 1:M
        indexrange[i] = i:M:ndata
        # label = rand(1:Int(32/M), length(indexrange[i]))
        fwks[i] = @spawnat i+1 Worker(data[:,indexrange[i]],i,M,hp)
    end
    fccs = Array{Future}(M)
    fvcs = Array{Future}(M)
    ccs = Array{ClusterCollection}(M)

    ## Init center
    c = Center(M)

    ## Evaluation
    like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
    lll = Array{Float64}(T)
    label = Array{Int}(ndata)
    maptable = Dict{Int,Int}()
    vi = Array{Float64}(T)
    runtime = zeros(Float64, T)
    bytes_transfer = zeros(Int,T)
    times_transfer = zeros(Int,T)
    count = Array{Int}(T)

    ## T iteration of CRP
    for iter in 1:T
        println("\niter = ", iter)

        logα0 = iter==1 ? logα-log(M) : logα

        ## Center send SuperClusterCollection to workers
        println("** Center -> Worker Communication")
        tic()
        @sync for wkid in 1:M
            vc = send_cluster_shallow(c,wkid)
            fvcs[wkid] = @spawnat wkid+1 vc

            ## Evaluate Bytes transfer
            for sc in vc
                bytes_transfer[iter] += (length(sc.ids) + 1 + hp.ds) * 8
            end
        end
        runtime[iter] += toc()
        times_transfer[iter] += M

        ## MCMC (parallel with local in algorithm but serial due to some problem in implementation)
        if TT > 0
            println("** MCMC")
            calc_table!(c,logα,hp)
            pooled_consolidation(c,TT,logα,hp)
        end

        ## Local (recv scc -> addother -> crp -> rmother -> send)
        println("** Local Iteration")
        tic()
        @sync for wkid in 1:M
            fwks[wkid] = @spawnat wkid+1 local_iteration!(fetch(fwks[wkid]),fetch(fvcs[wkid]),hp,logα0)
        end
        @sync for wkid in 1:M
            fccs[wkid] = @spawnat wkid+1 send_cluster(fetch(fwks[wkid]))
        end
        runtime[iter] += toc()

        ## Worker send Cluster Collection to Center
        println("** Worker -> Center Communication")
        tic()
        for wkid in 1:M
            ccs[wkid] = fetch(fccs[wkid])
            ## Evaluate Bytes transfer
            bytes_transfer[iter] += length(ccs[wkid])*(2 + hp.ds)*8
            # println(Base.summarysize(ccs[wkid]))
        end
        times_transfer[iter] += M
        runtime[iter] += toc()

        println("** Progressive Merge")
        tic()
        for wkid in 1:M
            recv_cluster_prog!(c,wkid,ccs[wkid],logα,hp)
            # recv_cluster_pool!(c,wkid,fetch(fccs[wkid]))
        end
        runtime[iter] += toc()

        eval && begin
        ### Evaluation
        ## 1. Collect labels and calc vi
        for wkid in 1:M
            label[indexrange[wkid]] = @fetchfrom wkid+1 send_label_shallow(fetch(fwks[wkid]))
        end
        empty!(maptable)
        for (sid,sc) in c.supercc, subid in sc.ids
            maptable[subid] = sid
        end
        for i in 1:ndata
            label[i] = maptable[label[i]]
        end
        vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(label),label)
        println("\n vi = ", vi[iter])

        ## 2. Collect models and calc likelihood
        lll[iter] = like_data + likelihood_cluster(send_cluster_shallow(c), logα, hp)
        println(" loglikelihood = ", lll[iter])
        end
        println(" bytes_transfer = ", bytes_transfer[iter])
        report_cluster_size(c)
        count[iter] = length(c.supercc)
    end

    println()
    println("vi = ", vi)
    println("loglikelihood = ", lll)
    println("runtime = ", runtime)
    println("bytes_transfer = ", bytes_transfer)
    println("times_transfer = ", times_transfer)

    return lll, vi, runtime, count, bytes_transfer, times_transfer
end


function evalAsyncRepeat(R::Int, M::Int, T::Int, TT::Int, dataset::Symbol, eval::Bool=true)
    lll = Array{Float64}(T,R)
    vi = Array{Float64}(T,R)
    runtime = zeros(Float64, T, R)
    count = Array{Float64}(T,R)
    bytes_transfer = Array{Float64}(T,R)
    times_transfer = Array{Float64}(T,R)

    for r in 1:R
        println("\nRepeat = ", r)
        lll[:,r], vi[:,r], runtime[:,r], count[:,r], bytes_transfer[:,r], times_transfer[:,r] = evalAsyncDistribute(M,T,TT,dataset)
    end

    lll_mean = mean(lll,2)
    vi_mean = mean(vi,2)
    runtime_mean = mean(runtime,2)
    count_mean = mean(count,2)
    cum_runtime_mean = cumsum(vcat(0,runtime_mean))
    bytes_transfer_mean = mean(bytes_transfer,2)
    times_transfer_mean = mean(times_transfer,2)

    lll_std = std(lll,2)
    vi_std = std(vi,2)
    runtime_std = std(runtime,2)
    count_std = std(count,2)
    bytes_transfer_std = std(bytes_transfer,2)
    times_transfer_std = std(times_transfer,2)

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
    println("times_transfer_mean = ", times_transfer_mean)
    println("times_transfer_std = ", times_transfer_std)
end
