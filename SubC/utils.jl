
function sample_parameter(N::Matrix{Int}, ss::Array{Float64,3}, logα::Float64, hp::HyperParameter)
   # N -- size K x 3, col 1 for regular, 2 for left, 3 for right
   # ss -- size hp.ds x K x 3

   alpha = exp(logα)
   K = size(N,1)
   assert(size(N,2) == 3)
   assert(size(ss,1) == hp.ds)
   assert(size(ss,2) == K)
   assert(size(ss,3) == 3)

   π = Array{Float64}(K,3)
   θ = Array{Float64}(hp.ds,K,3)

   # sample regular π
   tmp = rand(Distributions.Dirichlet(vcat(vec(N[:,1]),alpha)+eps()))
   π[:,1] = tmp[1:end-1]

   # sample sub π
   for k in 1:K
      π[k,2:3] = rand(Distributions.Dirichlet([N[k,2]+alpha/2, N[k,3]+alpha/2]))
   end

   # sample regular θ
   for k in 1:K, s in 1:3
      post = posterior(N[k,s], ss[:,k,s], hp)
      θ[:,k,s] = rand(post)
   end

   π .= log.(π)

   return π, θ #logπ
end


function sample_assignment(x::AbstractArray{Tv,2} where Tv,
   logπ::Matrix{Float64}, θ::Array{Float64,3},
   logα::Float64, hp::HyperParameter)

   K = size(logπ,1)
   ndata = size(x, 2)
   assert(size(logπ,2) == 3)
   assert(size(θ,1) == hp.ds)
   assert(size(θ,2) == K)
   assert(size(θ,3) == 3)

   # likelihoods = Array{typeoflikelihood(hp)}(K,3)
   # for k in 1:K, s in 1:3
   #    likelihoods[k,s] = likelihood(θ[:,k,s], hp)
   # end

   z = Array{Int}(size(x,2), 2)
   N = zeros(Int, K, 3)
   ss = zeros(Float64, hp.ds, K, 3)

   prob = Vector{Float64}(K)
   probs = Vector{Float64}(2)
   for i in 1:ndata
      # xi = x[:,i]
      xi = view(x,:,i)

      for k in 1:K
         # l = modify_likelihood(likelihoods[k,1], xi)
         # prob[k] = logπ[k,1] + Distributions.logpdf(l, xi)
         prob[k] = logπ[k,1] + loglikelihood(xi, θ[:,k,1], hp)
      end
      m = maximum(prob)
      prob .= exp.(prob.-m)

      k = mysample(prob)
      z[i,1] = k
      N[k,1] += 1
      ss[:,k,1] .+= xi

      for s in 2:3
         # l = modify_likelihood(likelihoods[k,s], xi)
         # probs[s-1] = logπ[k,s] + Distributions.logpdf(l, xi)
         probs[s-1] = logπ[k,s] + loglikelihood(xi, θ[:,k,s], hp)
      end
      m = maximum(probs)
      probs .= exp.(probs.-m)

      s = mysample(probs)
      z[i,2] = s
      N[k,s+1] += 1
      ss[:,k,s+1] .+= xi
   end

   return z, N, ss
end


function remove_empty(N::Matrix{Int}, ss::Array{Float64,3})
   K = 0
   for k in 1:size(N,1)
      N[k,1] > 0 && (K += 1)
   end

   N2 = Array{Int}(K,3)
   ss2 = Array{Float64}(size(ss,1),K,3)

   cur = 0
   for k in 1:size(N,1)
      N[k,1] > 0 || continue
      cur += 1
      N2[cur,:] = N[k,:]
      ss2[:,cur,:] = ss[:,k,:]
   end
   assert(cur==K)
   assert(sum(N)==sum(N2))
   assert(sum(ss)-sum(ss2)<eps(Float32))

   return N2, ss2
end


function two_component(data::AbstractArray{Tv,2} where Tv, hp::HyperParameter)
   ndata = size(data,2)
   z = Array{Int}(ndata)

   logπ = rand(2)
   θ = rand(hp.ds, 2)

   # likelihoods = Array{typeoflikelihood(hp)}(2)

   N = [ndata,ndata]
   ss = rand(hp.ds, 2)

   lli = Array{Float64}(2)
   ll_old = Inf
   ll = Inf

   prob = Array{Float64}(2)
   for iter in 1:10
      ll_old = ll
      ll = 0.0

      logπ = rand(Distributions.Dirichlet(N+eps()))
      logπ .= log.(logπ)

      # for k in 1:2
      #    θ[:,k] = rand(posterior(N[k], ss[:,k], hp))
      #    likelihoods[k] = likelihood(θ[:,k], hp)
      # end

      for i in 1:ndata
         # @time xi = data[:,i]
         xi = view(data,:,i)
         for k in 1:2
            # l = modify_likelihood(likelihoods[k], xi)
            # lli[k] = Distributions.logpdf(l, xi)
            # prob[k] = logπ[k] + lli[k]
            lli[k] = loglikelihood(xi, θ[:,k], hp)
            prob[k] = logπ[k] + lli[k]
         end
         m = maximum(prob)
         prob .= exp.(prob.-m)

         z[i] = mysample(prob)

         ll += lli[z[i]]
      end

      fill!(N, 0)
      fill!(ss, 0)
      for i in 1:ndata
         N[z[i]] += 1
         ss[:,z[i]] .+= data[:,i]
      end

      iter > 2 && ll < 0.95 * ll_old && break
   end

   assert(sum(N) == ndata)
   # assert(sum(abs,ss) - sum(abs,data) < eps(Float32))
   return z, N, ss
end


function split_subcluster(data::AbstractArray{Tv,2} where Tv,
   z::Matrix{Int}, N::Matrix{Int}, ss::Array{Float64,3},
   logα::Float64, hp::HyperParameter, T::Int)

   ndata = size(data, 2)
   assert(size(z,1) == ndata)
   K = size(N,1)
   assert(size(N,2) == 3)
   assert(size(ss,1) == hp.ds)
   assert(size(ss,2) == K)
   assert(size(ss,3) == 3)
   naccept = 0
   npropose = 0

   N = cat(1, N, zeros(Int,T,3))
   ss = cat(2, ss, zeros(Float64,hp.ds,T,3))

   index1 = Array{Int}(0)
   index2 = Array{Int}(0)
   for iter in 1:T
      # 1:K split, else merge
      # same probability for all possible merge and split
      ks = rand(1:K*(K+3)>>1)

      ks > K && continue  # merge is auto rejected
      N[ks,2] > 1 && N[ks,3] > 1 || continue  # skip small clusters

      npropose += 1

      c1 = Cluster(N[ks,1],ss[:,ks,1])
      c2 = Cluster(N[ks,2],ss[:,ks,2])
      c3 = Cluster(N[ks,3],ss[:,ks,3])

      logH = logα + lgamma(N[ks,2]) + lgamma(N[ks,3]) - lgamma(N[ks,1]) + _b(c2, hp) + _b(c3, hp) - _b(c1, hp)

      if rand() < min(1, exp(logH))
         naccept += 1
         # ks, left -> ks; ks, right -> K+1
         K += 1

         resize!(index1, 0)
         resize!(index2, 0)
         for i in 1:ndata
            z[i,1]==ks && z[i,2]==1 && push!(index1, i)
            z[i,1]==ks && z[i,2]==2 && push!(index2, i)
         end

         z[index2,1] = K
         N[ks,1] = N[ks,2]
         N[K,1] = N[ks,3]
         ss[:,ks,1] = ss[:,ks,2]
         ss[:,K,1] = ss[:,ks,3]

         println("two start")
         f1 = @spawn two_component(data[:,index1], hp)
         f2 = @spawn two_component(data[:,index2], hp)
         z1, N1, ss1 = fetch(f1)
         z2, N2, ss2 = fetch(f2)
         println("two finish")

         z[index1,2] = z1
         z[index2,2] = z2
         N[ks,2:3] = N1
         N[K,2:3] = N2
         ss[:,ks,2:3] = ss1
         ss[:,K,2:3] = ss2
      end

      assert(sum(N[:,1]) == ndata)
      assert(N[:,1] == N[:,2] + N[:,3])
      assert(sum(abs2, ss[:,:,1]-ss[:,:,2]-ss[:,:,3]) < eps(Float32))
   end

   println("#propose = ", npropose)
   println("#accept = ", naccept)
   return N[1:K,:], ss[:,1:K,:]
end
