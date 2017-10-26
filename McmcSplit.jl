
global q1 = 1

## Propose a split of a supercluster using restricted progressive consolidation
function propose_split(sc::SuperCluster, subcc::SubClusterCollection, logα::Float64, hp::HyperParameter)
    cum_log_prob::Float64 = 0
    cum_sc = [SuperCluster(hp),SuperCluster(hp)]
    log_prob = Array{Float64}(2)
    cm = Cluster(hp)
    K = 0

    for id in sc.ids
        sub = subcc[id]
        if K == 0
            K = 1
            add_sub!(cum_sc[1],sub,id)
        else
            log_prob[1] = logρ!(cum_sc[1], sub, cm, logα, hp) * q1
            log_prob[2] = K==1 ? 0 : logρ!(cum_sc[2], sub, cm, logα, hp) * q1

            log_prob -= maximum(log_prob)
            prop_prob = exp.(log_prob)

            u = mysample(prop_prob)

            cum_log_prob += log_prob[u]-log(sum(prop_prob))
            add_sub!(cum_sc[u],sub,id)

            K==1 && u==2 && (K=2)
        end
    end
    NOASSERT || assert(union(cum_sc[1].ids,cum_sc[2].ids) == sc.ids)
    assert(cum_sc[1].n+cum_sc[2].n == sc.n)
    NOASSERT || assert(sumabs2(cum_sc[1].ss+cum_sc[2].ss-sc.ss) < eps(Float32))
    assert(K==1 || K==2)
    assert(cum_log_prob <= 0)
    return K==1, cum_log_prob, cum_sc
end


## Probability that (A+B) will split to A and B
function log_split_prob(sc::SuperCluster, subids::Set{Int}, subcc::SubClusterCollection, logα::Float64, hp::HyperParameter)
    assert(issubset(subids, sc.ids))
    cum_log_prob::Float64 = 0
    cum_sc = [Cluster(hp),Cluster(hp)]
    log_prob = Array{Float64}(2)
    cm = Cluster(hp)
    K = 0
    thisid = -1  # for cluster in subids
    thatid = -1  # for cluster in sc.ids \ subids

    for id in sc.ids
        sub = subcc[id]
        if K == 0
            K = 1
            add_stat!(cum_sc[1],sub)
            isin = in(id,subids)
            thisid = isin ? 1 : 2
            thatid = isin ? 2 : 1
        else
            log_prob[1] = logρ!(cum_sc[1], sub, cm, logα, hp) * q1
            log_prob[2] = K==1 ? 0 : logρ!(cum_sc[2], sub, cm, logα, hp) * q1

            log_prob -= maximum(log_prob)

            u = in(id,subids) ? thisid : thatid

            cum_log_prob += log_prob[u]-log(exp(log_prob[1])+exp(log_prob[2]))
            add_stat!(cum_sc[u],sub)

            K==1 && u==2 && (K=2)
        end
    end
    assert(cum_log_prob <= 0)
    return cum_log_prob
end


## Probability that the Pool algorithm results in a single cluster, Eq(7) in paper
## β in Eq (7) in the paper
function log_no_split_prob(sc::SuperCluster, subcc::SubClusterCollection, logα::Float64, hp::HyperParameter)
    log_β::Float64 = 0

    logp1::Float64 = 0
    logp2::Float64 = 0

    cum_sc = Cluster(hp)
    cm = Cluster(hp)
    K = 0

    for id in sc.ids
        sub = subcc[id]
        if K == 0
            K = 1
        else
            logp1 = logρ!(cum_sc, sub, cm, logα, hp) * q1
            if logp1 <= 0.
                logp2 = 0.
            else
                logp2 = -logp1
                logp1 = 0.
            end
            #log(exp(logp1)/(exp(logp1)+exp(logp2)))
            log_β += logp1-log(exp(logp1)+exp(logp2))
        end
        add_stat!(cum_sc,sub)
    end
    assert(cum_sc.n == sc.n)
    NOASSERT || assert(sumabs2(cum_sc.ss - sc.ss) < eps(Float32))
    assert(log_β <= 0)
    return log_β
end
