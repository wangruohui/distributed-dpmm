
## Calculate sufficient statistics given samples
function add_ss!(ss::Vector{Float64}, x::Vector, hp::IsotropicGaussian)
    assert(length(ss)==length(x))
    @simd for d in eachindex(x)
        @inbounds ss[d] += x[d]
    end
end

function sub_ss!(ss::Vector{Float64}, x::Vector, hp::IsotropicGaussian)
    assert(length(ss)==length(x))
    @simd for d in eachindex(x)
        @inbounds ss[d] -= x[d]
    end
end

function loghx(x::Matrix, hp::IsotropicGaussian)
    # x can be vector or matrix
    N = size(x,2)
    d = hp.dd
    σ = hp.σ
    # -sumabs2(x)/2σ^2 - N*d*(log(σ)+log(2π)/2)
    -sum(abs2,x)/2σ^2 - N*d*(log(σ)+log(2π)/2)
end

## log partition function b
function _b(c::AbstractCluster, hp::IsotropicGaussian)
    κ = hp.κ0 + hp.c * c.n
    # sumabs2(c.ss)/2κ/hp.σ^4 + hp.dim/2*log(2π/κ)
    sum(abs2,c.ss)/2κ/hp.σ^4 + hp.dd/2*log(2π/κ)
end

## when c is empty
_b(hp::IsotropicGaussian) = hp.dd/2*log(2π/hp.κ0)

## b(x+C)-b(C)
function diff_b(x::Vector, c::DataCluster, hp::IsotropicGaussian)
    β = c.ss
    κ = hp.κ0 + hp.c * c.n
    t = κ + hp.c
    # (sumabs2(x)/2 - sumabs2(β)*hp.c/2κ + dot(β,x))/t/hp.σ^4 + hp.dim/2*log(κ/t)
    (sum(abs2,x)/2 - sum(abs2,β)*hp.c/2κ + dot(β,x))/t/hp.σ^4 + hp.dd/2*log(κ/t)
end

## when c is empty
function diff_b(x::Vector, hp::IsotropicGaussian)
    t = hp.κ0 + hp.c
    # sumabs2(x)/2t/hp.σ^4 + hp.dim/2*log(hp.κ0/t)
    sum(abs2,x)/2t/hp.σ^4 + hp.dd/2*log(hp.κ0/t)
end
