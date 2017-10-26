
## Calculate sufficient statistics given samples
function add_ss!(ss::Vector{Float64}, x::SparseVector{Int,Int}, hp::MultinomialDir)
    @simd for i in eachindex(x.nzind)
        @inbounds ss[x.nzind[i]] += x.nzval[i]
    end
end

function sub_ss!(ss::Vector{Float64}, x::SparseVector{Int,Int}, hp::MultinomialDir)
    @simd for i in eachindex(x.nzind)
        @inbounds ss[x.nzind[i]] -= x.nzval[i]
    end
end

function loghx(X::SparseVector{Int}, hp::MultinomialDir)
    N = sum(X)
    return sum(lgamma, 1+N) - sum(lgamma, 1+X.nzval)
end

function loghx(X::SparseMatrixCSC{Int,Int}, hp::MultinomialDir)
    # each column is a document
    N = sum(X,1)
    return sum(lgamma, 1+N) - sum(lgamma, 1+X.nzval)
end

## log partition function b
function _b(c::AbstractCluster, hp::MultinomialDir)
    slg = 0.0
    for i in eachindex(hp.γ)
        @inbounds slg += lgamma(hp.γ[i] + c.ss[i])
    end
    return slg - lgamma(sum(hp.γ) + sum(c.ss))
end

## when c is empty
_b(hp::MultinomialDir) = sum(lgamma,hp.γ) - lgamma(sum(hp.γ))

## b(x+C)-b(C)
function diff_b(x::SparseVector{Int,Int}, c::DataCluster, hp::MultinomialDir)
    sl = 0.0
    @simd for i in eachindex(x.nzind)
        @inbounds ind = x.nzind[i]
        @inbounds xi = x.nzval[i]
        @inbounds γi = hp.γ[ind]
        @inbounds ssi = c.ss[ind]
        sl += lgamma(xi+γi+ssi) - lgamma(γi+ssi)
    end
    s1 = sum(hp.γ)+sum(c.ss)
    s2 = sum(x.nzval)
    return sl - lgamma(s1+s2) + lgamma(s1)
end

## when c is empty
function diff_b(x::SparseVector{Int,Int}, hp::MultinomialDir)
    sl = 0.0
    @simd for i in eachindex(x.nzind)
        @inbounds ind = x.nzind[i]
        @inbounds xi = x.nzval[i]
        @inbounds γi = hp.γ[ind]
        sl += lgamma(xi+γi) - lgamma(γi)
    end
    s1 = sum(hp.γ)
    s2 = sum(x.nzval)
    return sl - lgamma(s1+s2) + lgamma(s1)
end
