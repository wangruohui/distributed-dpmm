using Clustering
using Distributions
using HDF5
using JLD

@everywhere begin
    include("../HyperParameter.jl")
    include("../Cluster.jl")
    include("../Gaussian.jl")
    include("../Multinomial.jl")
    include("../ExperimentSetting.jl")
    include("../mysample.jl")
    include("../Evaluation.jl")
    include("utils.jl")
    include("probability.jl")
end

function evalSliceMRDistributed(M::Int, T::Int, KINIT::Int, dataset::Symbol)
    assert(M <= nworkers())

    ## Import Experiment Setting
    data,gt,logα,hp = get_exp_setting(dataset)
    ndata = length(gt)

    ## Allocation & Initialization
    fwks = Array{Future}(M)
    @sync for m in 1:M
        fwks[m] = @spawnat m+1 SliceWorker(data[:,m:M:end])
    end

    mstar = 1
    kstar = 1
    K = KINIT
    beta = rand(K)
    beta ./= sum(beta)
    ustar = minimum(beta)
    phi = rand(hp.ds, K)
    maptable = [1]

    # evaluation
    like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
    lll = Array{Float64}(T)
    label = Array{Int}(ndata)
    vi = Array{Float64}(T)
    runtime = zeros(Float64, T)
    bytes_transfer = zeros(Int,T)
    times_transfer = zeros(Int,T)
    count = Array{Int}(T)

    # T iterations
    for iter in 1:T
        println("\niter = ", iter)

        println("Local")
        tic()
        @sync for m in 1:M
            fwks[m] = @spawnat m+1 slicelocal!(fetch(fwks[m]), maptable, ustar, m==mstar, kstar, beta, phi, logα, hp)
        end
        runtime[iter] += toc()
        bytes_transfer[iter] += 8 * M * (length(beta) + length(phi) + length(maptable) + 3)
        times_transfer[iter] += M

        println("Global")
        tic()
        ## accumulate
        n_full = Array{Int}(M, K)
        n = zeros(Int, K)
        psi = zeros(Float64, hp.ds, K)

        for m in 1:M
            f = @spawnat m+1 emit_ss(fetch(fwks[m]), K, hp)
            n1, psi1 = fetch(f)
            n_full[m,:] = n1
            n .+= n1
            psi .+= psi1
        end
        bytes_transfer[iter] += 8 * M * K * (1+hp.ds)
        times_transfer[iter] += M

        # active component
        nzind = find(n.!=0)
        n = n[nzind]
        psi = psi[:,nzind]
        n_full = n_full[:, nzind]
        maptable = zeros(Int, nzind[end])
        maptable[nzind] = collect(1:length(nzind))

        ustar, mstar, kstar, beta, phi = sliceglobal(n, psi, n_full, logα, hp)
        K = length(beta)

        runtime[iter] += toc()

        # Eval
        tic()
        count[iter] = K
        println(sort(n))
        for m in 1:M
            f = @spawnat m+1 emit_z_shallow(fetch(fwks[m]))
            label[m:M:end] = fetch(f)
        end
        vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(label),label)

        cc = Vector{Cluster}(0)
        for k in 1:length(n)
            push!(cc, Cluster(n[k], psi[:,k]))
        end
        lll[iter] = like_data + likelihood_cluster(cc, logα, hp)
        println("eval ",toc())

        println()
        println(" vi = ", vi[iter])
        println(" lll = ", lll[iter])
        println(" runtime = ", runtime[iter])
        println(" count = ", count[iter])
        println(" bytes_transfer = ", bytes_transfer[iter])
    end

    println()
    println(" vi = ", vi)
    println(" loglikelihood = ", lll)
    println(" runtime = ", runtime)
    println(" count = ", count)
    println(" bytes_transfer = ", bytes_transfer)
    println(" times_transfer = ", times_transfer)

    return lll, vi, runtime, count, bytes_transfer
end



function evalSliceMRDistributedRepeat(R::Int, M::Int, T::Int, KINIT::Int, dataset::Symbol)

    lll = Array{Float64}(T, R)
    vi = Array{Float64}(T, R)
    runtime = zeros(Float64, T, R)
    count = Array{Float64}(T, R)
    bytes_transfer = Array{Float64}(T, R)

    for r in 1:R
        println("Repeat = ", r)
        lll[:,r], vi[:,r], runtime[:,r], count[:,r], bytes_transfer[:,r] = evalSliceMRDistributed(M,T,KINIT,dataset)
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
