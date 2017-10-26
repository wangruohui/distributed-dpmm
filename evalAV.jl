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


## function extension of Worker
@everywhere function send_dcc_size(w::Worker)
   cids = Array{Int}(0)
   sizes = Array{Int}(0)
   for (id,dc) in w.dcc
      push!(cids,id)
      push!(sizes,dc.n)
   end
   return cids, sizes
end

@everywhere function send_dcc_shallow!(w::Worker, id::Int)
   N = size(w.data,2)
   id_send = sort(collect(w.dcc[id].ids))
   id_remain = deleteat!(collect(1:N), id_send)
   M = length(id_send)

   cc_send = Cluster(w.dcc[id].n, w.dcc[id].ss)
   data_send = w.data[:,id_send]

   delete!(w.dcc,id)

   w.data = w.data[:,id_remain]
   deleteat!(w.label,id_send)

   for dc in values(w.dcc)
      empty!(dc.ids)
   end

   for (i,l) in enumerate(w.label)
      union!(w.dcc[l].ids,i)
   end

   return data_send, cc_send, id_send, N
end

@everywhere function recv_dcc!(w::Worker, data_recv::AbstractMatrix, cc::Cluster, cid::Int)
   data_ids = size(w.data,2) + (1:size(data_recv,2))
   w.data = hcat(w.data,data_recv)
   resize!(w.label,data_ids.stop)
   w.label[data_ids] = cid
   assert(cid!=0)
   assert(!haskey(w.dcc,cid))
   w.dcc[cid] = DataCluster(Set{Int}(data_ids),cc.n,cc.ss)
   while w.nextid <= cid
      w.nextid += w.idstep
   end
   return w
end

@everywhere report_nclusters(w::Worker) = length(w.dcc)

function evalAVparallel(M::Int, T::Int, TT::Int, KINIT::Int, dataset::Symbol)
   # srand(10)
   assert(M<=nworkers())

   ## Import Experiment Setting
   data,gt,logα,hp = get_exp_setting(dataset)
   ndata = length(gt)

   logα0 = logα-log(M)

   ## Init workers
   indexrange = Array{Vector{Int}}(M)
   fwks = Array{Future}(M)
   @sync for i in 1:M
      indexrange[i] = collect(i:M:ndata)
      fwks[i] = @spawnat i+1 Worker(data[:,indexrange[i]],i,M,hp,KINIT)
   end
   # fccs = Array(Future,M)
   # fvcs = Array(Future,M)
   # ccs = Array(ClusterCollection,M)

   ## Init center
   c = Center(M)

   ## Evaluation
   like_data = loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
   lll = Array{Float64}(T)
   label = Array{Int}(ndata)
   vi = Array{Float64}(T)
   runtime = zeros(Float64, T)
   comm_bytes = zeros(Int,T)
   comm_times = zeros(Int,T)
   count = Array{Int}(T)

   ## T iteration of CRP
   for iter in 1:T
      println("\niter = ", iter)

      ## Local Inference
      println("Local Inference")
      tic()
      @sync for wkid in 1:M
         fwks[wkid] = @spawnat wkid+1 crp!(fetch(fwks[wkid]),hp,logα0)
      end
      runtime[iter] += toc()

      ## Global Inference
      println("Global Inference")
      tic()
      println("Compute A_ij")
      mat_a = sparse(Vector{Int}(),Vector{Int}(),Vector{Int}(),ndata,ndata)
      allcids = Array{Int}(0)
      allwkids = Array{Int}(0)
      allsizes = Array{Int}(0)
      for wkid in 1:M
         cids, sizes = @fetchfrom wkid+1 send_dcc_size(fetch(fwks[wkid]))
         for sz in sizes
            mat_a[sz,wkid] += 1
            push!(allsizes,sz)
         end
         for cid in cids
            push!(allcids, cid)
            push!(allwkids, wkid)
         end
         comm_bytes[iter] += 8*2*length(cids)
         comm_times[iter] += 1
      end
      runtime[iter] += toc()

      tic()
      println("Moving samples")
      rej_count = 0
      for tt in 1:TT
         kidx = rand(1:length(allcids))
         k = allcids[kidx]
         oj = allwkids[kidx]
         j = rand(1:M)

         j==oj && continue

         i = allsizes[kidx]

         loga1 = lgamma(mat_a[i,oj]+1)

         mat_a[i,oj] -= 1
         mat_a[i,j] += 1

         loga2 = lgamma(mat_a[i,j]+1)

         ratio = exp(min(0,loga1-loga2))

         if rand() < ratio
            # accept
            allwkids[kidx] = j
            data_trans, cc_trans, id_send,Nlocal = @fetchfrom oj+1 send_dcc_shallow!(fetch(fwks[oj]), k)
            fwks[j] = @spawnat j+1 recv_dcc!(fetch(fwks[j]), data_trans, cc_trans, k)
            comm_bytes[iter] += 8*length(data_trans)
            comm_bytes[iter] += 8*(2+hp.ds)
            comm_bytes[iter] += 8*length(id_send)+8
            comm_times[iter] += 1
            indexrange[j] = vcat(indexrange[j],indexrange[oj][id_send])
            id_remain = deleteat!(collect(1:Nlocal), id_send)
            indexrange[oj] = indexrange[oj][id_remain]
            wait(fwks[j])
         else
            # reject
            rej_count += 1
            mat_a[i,oj] += 1
            mat_a[i,j] -= 1
         end
      end
      runtime[iter] += toc()
      println("reject count = ", rej_count)

      ### Merge
      println("Local Progressive Merge")
      tic()
      @sync for wkid in 1:M
         fwks[wkid] = @spawnat wkid+1 begin
         wk = fetch(fwks[wkid])
         ct = Center(1)
         ccs = send_cluster(wk)
         recv_cluster_prog!(ct,1,ccs,logα0,hp)
         vc = send_cluster_shallow(ct)
         recv_cluster!(wk, vc) end
      end
      runtime[iter] += toc()


      ### Evaluation
      ## 1. Collect labels and calc vi
      for wkid in 1:M
         label[indexrange[wkid]] = @fetchfrom wkid+1 send_label_shallow(fetch(fwks[wkid]))
      end
      vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(label),label)
      println("\n vi = ", vi[iter])

      ## 2. Collect models and calc likelihood
      lll[iter] = like_data
      for wkid in 1:M
         cc = @fetchfrom wkid+1 send_cluster(fetch(fwks[wkid]))
         lll[iter] += likelihood_cluster(collect(values(cc)), logα, hp)
      end
      println(" loglikelihood = ", lll[iter])

      ### 3. count of clusters
      count[iter] = 0
      for wkid in 1:M
         a = @fetchfrom wkid+1 report_nclusters(fetch(fwks[wkid]))
         a == 0 && println(wkid, " is empty")
         count[iter] += a
      end
      println(" count = ", count[iter])
   end

   println()
   println("vi = ", vi)
   println("loglikelihood = ", lll)
   println("runtime = ", runtime)
   println("count = ", count)
   println("comm_bytes = ", comm_bytes)
   println("comm_times = ", comm_times)

   return lll, vi, runtime, count, comm_bytes, comm_times
end


function evalAVrepeat(R::Int, M::Int, T::Int, TT::Int, dataset::Symbol)
   lll = Array{Float64}(T,R)
   vi = Array{Float64}(T,R)
   runtime = zeros(Float64, T, R)
   count = Array{Float64}(T,R)
   comm_bytes = Array{Float64}(T,R)
   comm_times = Array{Float64}(T,R)

   for r in 1:R
      println("\n Repeat = ", r)
      lll[:,r], vi[:,r], runtime[:,r], count[:,r], comm_bytes[:,r], comm_times[:,r] = evalAVparallel(M,T,TT,dataset)
      sleep(3)
   end

   lll_mean = mean(lll,2)
   vi_mean = mean(vi,2)
   runtime_mean = mean(runtime,2)
   count_mean = mean(count,2)
   cum_runtime_mean = cumsum(vcat(0,runtime_mean))
   comm_bytes_mean = mean(comm_bytes,2)
   comm_times_mean = mean(comm_times,2)

   lll_std = std(lll,2)
   vi_std = std(vi,2)
   runtime_std = std(runtime,2)
   count_std = std(count,2)
   comm_bytes_std = std(comm_bytes,2)
   comm_times_std = std(comm_times,2)

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
   println("comm_bytes_mean = ", comm_bytes_mean)
   println("comm_bytes_std = ", comm_bytes_std)
   println("comm_times_mean = ", comm_times_mean)
   println("comm_times_std = ", comm_times_std)
end
