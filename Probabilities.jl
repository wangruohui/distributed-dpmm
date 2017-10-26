
## calculate probability vector for CRP
function calc_prob!(prob::Vector{Float64}, ids::Vector{Int},
                        cc::DataClusterCollection, x::AbstractVector,
                        hp::HyperParameter, logα::Float64)
    empty!(prob)
    empty!(ids)
    for id in keys(cc)
        push!(ids, id)
        push!(prob, log(cc[id].n) + diff_b(x, cc[id], hp))
    end
    push!(ids, -1)
    push!(prob, logα + diff_b(x, hp))
    max_log_prob = maximum(prob)
    for i in eachindex(prob)
        @inbounds prob[i] = exp(prob[i]-max_log_prob)
    end
end

## log merge split ratio
function logρ(c1::AbstractCluster, c2::AbstractCluster, logα::Float64, hp::HyperParameter)
    # cm is not preallocated
    cm = Cluster(hp)
    logρ!(c1, c2, cm, logα, hp)
end

function logρ!(c1::AbstractCluster, c2::AbstractCluster, cm::AbstractCluster,
                logα::Float64, hp::HyperParameter)
    # cm is preallocated but not calculated
    # cm will become c1+c2, there is no need to calculate c1+c2 again
    cm.n = c1.n+c2.n
    @simd for d in eachindex(cm.ss)
        @inbounds cm.ss[d] = c1.ss[d] + c2.ss[d]
    end
    logρ(c1, c2, cm, logα, hp)
end

function logρ(c1::AbstractCluster, c2::AbstractCluster, cm::AbstractCluster, logα::Float64, hp::HyperParameter)
    # cm is already calculated
    NOASSERT && assert(c1.n>0)
    NOASSERT && assert(c2.n>0)
    NOASSERT && assert(c1.n+c2.n==cm.n)
    a = -logα + lgamma(cm.n) - lgamma(c1.n) - lgamma(c2.n)
    b = _b(cm,hp) + _b(hp) - _b(c1,hp) - _b(c2,hp)
    return a+b
end
