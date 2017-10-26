
type SliceWorker
   y        :: AbstractMatrix    # data
   z        :: Vector{Int}
   u        :: Vector{Float64}
   function SliceWorker(data::AbstractMatrix)
      ndata = size(data, 2)
      y = deepcopy(data)
      z = ones(Int,ndata)
      u = Array{Float64}(ndata)
      new(y,z,u)
   end
end

function slicelocal!(w::SliceWorker, maptable::Vector{Int},
   ustar::Float64, sampleistar::Bool, kstar::Int, beta::Vector{Float64}, phi::Matrix{Float64},
   logα::Float64, hp::HyperParameter)

   y = w.y
   z = w.z
   u = w.u

   ##### change label
   for i in eachindex(z)
      z[i] = maptable[z[i]]
   end

   #### sample ui
   for i in eachindex(z)
      u[i] = rand() * (beta[z[i]] - ustar) + ustar
   end
   if sampleistar
      ikstar = Array{Int}(0)
      for i in eachindex(z)
         z[i] == kstar && push!(ikstar,i)
      end
      istar = rand(ikstar)
      u[istar] = ustar
   end

   #### sample zi
   ## 1. sort beta
   ix = sortperm(beta, rev=true)
   beta = beta[ix]
   phi = phi[:,ix]

   ## 2. construct distribution
   K = length(beta)
   #  G = Array{Distributions.MvNormal}(K)
   #  for k in 1:K
   #      G[k] = Distributions.MvNormal(phi[:,k],hp.σ)
   #  end

   ## 3. sample z[i]
   prob = Array{Float64}(0)
   for i in eachindex(z)
      resize!(prob,0)
      yi = y[:,i]
      for k in 1:K
         beta[k] < u[i] && break
         # push!(prob, Distributions.logpdf(G[k], yi))
         logpdf = loglikelihood(yi, phi[:,k], hp)
         push!(prob, logpdf)
      end
      m = maximum(prob)
      prob .= exp.(prob.-m)
      z[i] = mysample(prob)
   end

   return w
end

function emit_ss(w::SliceWorker, K::Int, hp::HyperParameter)
   z = w.z
   y = w.y

   n = zeros(Int, K)
   psi = zeros(Float64, hp.ds, K)
   for i in eachindex(z)
      zi = z[i]
      n[zi] += 1
      psi[:,zi] += y[:,i]
   end

   return n, psi
end

function emit_z_shallow(w::SliceWorker)
   return w.z
end

function acc_ss(fss::Vector{Future})
   N, psi = fetch(fss[1])
   N_full = Array{Int}(length(fss), length(N))

   for m in 2:length(fss)
      N1, psi1 = fetch(fss[m])
      N .+= N1
      psi .+= psi1
   end

   return N, psi, N_full
end

function sliceglobal(n::Vector{Int}, psi::Matrix{Float64}, n_full::Matrix{Int},
   logα::Float64, hp::HyperParameter)

   # # active component
   # K = 0
   # maptable = zeros(Int,maximum(z))
   # for i in eachindex(z)
   #     zi = z[i]
   #     if maptable[zi] == 0
   #         K += 1
   #         maptable[zi] = K
   #     end
   # end
   # for i in eachindex(z)
   #     z[i] = maptable[z[i]]
   # end
   #
   # #### accumulate
   # n = SharedArray{Int}(K)
   # psi = SharedArray{Float64}(hp.ds, K)
   # for i in eachindex(z)
   #     zi = z[i]
   #     n[zi] += 1
   #     psi[:,zi] += y[:,i]
   # end
   K = length(n)
   #### sample beta
   a = vcat(n,exp(logα)) + eps()
   beta = rand(Distributions.Dirichlet(a))

   #### sample ustar istar
   b = Array{Float64}(K)
   u = Array{Float64}(K)
   for k in 1:K
      b[k] = rand(Distributions.Beta(1,n[k]+eps()))
      u[k] = beta[k] * b[k]
   end
   ustar, kstar = findmin(u)

   mstar = mysample(n_full[:,kstar])
   # ikstar = Array{Int}(0)
   # for i in eachindex(z)
   #     z[i] == kstar && push!(ikstar,i)
   # end
   # istar = rand(ikstar)

   #### Init new
   KSTAR = K
   B = Distributions.Beta(1,exp(logα))
   while beta[end] >= ustar
      KSTAR += 1
      nu = rand(B)
      push!(beta,beta[end]*(1-nu))
      beta[KSTAR] = beta[KSTAR] - beta[end]
      # println(beta)
   end
   ##! disgard beta star
   pop!(beta)

   #### sample all phi
   phi = Array{Float64}(hp.ds, KSTAR)
   for k in 1:K
      # σ0_post_sq = 1/(1/hp.σ0^2 + n[k]/hp.σ^2)
      # μ0_post = psi[:,k] / hp.σ^2 * σ0_post_sq
      # para_post_dist = Distributions.MvNormal(μ0_post, sqrt(σ0_post_sq))
      para_post_dist = posterior(n[k], psi[:,k], hp)
      phi[:,k] = rand(para_post_dist)
   end
   # phi[:,K+1:KSTAR] = rand(Distributions.MvNormal(hp.ds,hp.σ0), KSTAR-K)
   phi[:,K+1:KSTAR] = rand(posterior(hp), KSTAR-K)

   # betas = SharedArray{Float64}(length(beta))
   # betas[:] = beta[:]
   # phis = SharedArray{Float64}(hp.ds, KSTAR)
   # phis[:] = phi[:]

   return ustar, mstar, kstar, beta, phi
end
