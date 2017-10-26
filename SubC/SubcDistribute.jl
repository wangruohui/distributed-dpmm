using Distributions
using Clustering
using HDF5
using JLD

@everywhere begin
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

function evalSubcDistribute(M::Int, T::Int, TT::Int, KINIT::Int,dataset::Symbol)
   assert(M<=nworkers())
   # srand(10)

   ## Import Experiment Setting
   data,gt,logα,hp = get_exp_setting(dataset)
   # println("logα -= 1000")
   # logα -= 1000
   ndata = length(gt)

   ## Evaluation
   like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
   lll = Array{Float64}(T+1)
   vi = Array{Float64}(T+1)
   count = Array{Int}(T+1)
   runtime = Array{Float64}(T)
   bytes_transfer = zeros(Int,T)
   times_transfer = zeros(Int,T)

   ## Distribute data
   println("Distribute Data")
   fdata = Array{Future}(M)
   @sync for m in 1:M
      fdata[m] = @spawnat m+1 data[:,m:M:ndata]
   end
   fstat = Array{Future}(M)

   ## Init state
   println("Initialize")
   K = 20
   # z = Array{Int}(ndata, 2)
   # N = zeros(Int, K, 3)
   # ss = zeros(Float64, hp.ds, K, 3)
   # z[:,1] = rand(1:K)
   # z[:,2] = rand(1:2)
   # for i in 1:ndata
   #    xi = data[:,i]
   #
   #    k = z[i,1]
   #    N[k,1] += 1
   #    ss[:,k,1] .+= xi
   #
   #    s = z[i,2]
   #    N[k,s+1] += 1
   #    ss[:,k,s+1] .+= xi
   # end
   init_logπ = rand(K, 3)
   init_θ = rand(hp.ds, K, 3)
   for i in 1:size(init_θ,2), j in 1:size(init_θ,3)
     init_θ[:,i,j] ./= sum(init_θ[:,i,j])
   end
   @time _, N, ss = sample_assignment(data[:,1:round(Int,sqrt(ndata)):end], init_logπ, init_θ, logα, hp)
   N .*= round(Int,sqrt(ndata))
   ss .*= round(Int,sqrt(ndata))
   z = Array{Int}(ndata, 2)
   z[:,1] = rand(1:K)
   z[:,2] = rand(1:2)

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

      logπ, θ = sample_parameter(N, ss, logα, hp)

      @time @sync for m in 1:M
         fstat[m] = @spawnat m+1 sample_assignment(fetch(fdata[m]), logπ, θ, logα, hp)
      end
      bytes_transfer[iter] = M * 8 * (length(logπ) + length(θ))
      times_transfer[iter] += M

      fill!(N,0)
      fill!(ss,0)
      for m in 1:M
         zp, Np, ssp = fetch(fstat[m])
         z[m:M:ndata, :] = zp
         N .+= Np
         ss .+= ssp
      end
      bytes_transfer[iter] = 8 * (length(z) + M*(length(N)+length(ss)) )
      times_transfer[iter] += M

      assert(sum(N[:,1]) == ndata)
      # assert(sum(ss[:,:,1]) - sum(data) < sqrt(eps(typeof(data[1]))))
      assert(N[:,1] == N[:,2] + N[:,3])
      # assert(sum(abs2, ss[:,:,1]-ss[:,:,2]-ss[:,:,3]) < eps(Float32))

      @time N, ss = split_subcluster(data, z, N, ss, logα, hp, TT)
      N, ss = remove_empty(N, ss)

      runtime[iter] = toq()
      println(" runtime = ", runtime[iter])

      println(sort(N[:,1]))
   end

   println()
   println("vi = ", vi)
   println("loglikelihood = ", lll)
   println("runtime = ", runtime)
   println("count = ", count)
   println("bytes_transfer = ", bytes_transfer)
   println("times_transfer = ", times_transfer)

   return lll, vi, runtime, count, bytes_transfer
end



function evalSubcDistributedRepeat(R::Int, M::Int, T::Int, KINIT::Int, dataset::Symbol)

    lll = Array{Float64}(T, R)
    vi = Array{Float64}(T, R)
    runtime = zeros(Float64, T, R)
    count = Array{Float64}(T, R)
    bytes_transfer = Array{Float64}(T, R)

    for r in 1:R
        println("Repeat = ", r)
        lll[:,r], vi[:,r], runtime[:,r], count[:,r], bytes_transfer[:,r] = evalSubcDistribute(M,T,KINIT,dataset)
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
