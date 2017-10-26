
abstract type HyperParameter end

# dd -- dimension of data (x)
# ds -- dimension of sufficient statistics (ss)

# Isotropic Multiraviate Gaussian Distribution with zero Mean

# log(h(x)) = -xᵀx/2σ^2 - dim*log(σ) - (dim/2)*log(2π)
# ψ(x) = x
# η(μ) = μ/σ^2
# a(μ) = μᵀμ/2
# c = 1/(σ^2)
# β₀ = 0
# κ₀ = 1/(σ₀^2)
# b(β,κ) = βᵀβ/(2κσ^4) + (dim/2)*log(2π/κ)

immutable IsotropicGaussian <: HyperParameter
   dd  ::  Int
   ds  ::  Int
   σ0  ::  Float64
   σ   ::  Float64
   c   ::  Float64
   κ0  ::  Float64
   IsotropicGaussian(dim::Int,σ0::Real,σ::Real) = new(dim,dim,σ0,σ,1/σ^2,1/σ0^2)
end

# Multinomial Distribution with Dirichlet Prior

# Hyper Parameter for Dirichlet Prior: γ
# Parameter: p -- word frequency, k -- vocabulary size, n -- total words
# Variable: x -- word count

# log(h(x)) = log(n!/Πxᵢ!) = lgamma(1+n)-Σlgamma(1+xᵢ) <lgamma(1)=0>
# ψ(x) = x
# η(p) = log.(p)
# a(p) = 0
# β₀ = γ.-1
# b(β) = ΣlogΓ(βᵢ+1)-logΓ(Σ(βᵢ+1)) = ΣlogΓ(γᵢ+ssᵢ)-logΓ(Σ(γᵢ+ssᵢ))

immutable MultinomialDir <: HyperParameter
   dd   ::  Int
   ds   ::  Int
   γ    ::  Vector{Float64}
   #  logγ ::  Vector{Float64}
   MultinomialDir(γ::Vector{Float64}) = new(length(γ),length(γ),deepcopy(γ))
end
