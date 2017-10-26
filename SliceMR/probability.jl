

function posterior(n::Int, ss::Array{Float64}, hp::IsotropicGaussian)
    σ0_post_sq = 1/(1/hp.σ0^2 + n/hp.σ^2)
    μ0_post = ss / hp.σ^2 * σ0_post_sq
    return Distributions.MvNormal(μ0_post, sqrt(σ0_post_sq))
end

posterior(hp::IsotropicGaussian) = Distributions.MvNormal(hp.ds, hp.σ0)


posterior(n::Int, ss::Array{Float64}, hp::MultinomialDir) = Distributions.Dirichlet(ss+hp.γ)
posterior(hp::MultinomialDir) = Distributions.Dirichlet(hp.γ)


loglikelihood(x::AbstractVector{Float64}, θ::Array{Float64}, hp::IsotropicGaussian) = -sum(abs2,x-θ)/2/hp.σ^2

function loglikelihood(x::AbstractVector{Int}, θ::Array{Float64}, hp::MultinomialDir)
    s = 0.0
    for ind in eachindex(x)
        @inbounds xi = x[ind]
        xi == 0 && continue
        @inbounds γi = θ[ind]
        s += xi*log(γi) - lgamma(xi+1)
    end
    return s
end


# function loglikelihood(x::AbstractVector{Int}, θ::Array{Float64}, hp::MultinomialDir)
#     s = 0.0
#     @simd for i in eachindex(x.nzind)
#         @inbounds ind = x.nzind[i]
#         @inbounds xi = x.nzval[i]
#         @inbounds γi = θ[ind]
#         s += xi*log(γi) - lgamma(xi+1)
#     end
#     return s
# end
