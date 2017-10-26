
function likelihood_data(data::AbstractMatrix, logα::Float64, hp::HyperParameter)
   ndata = size(data,2)
   return loghx(data,hp) - lgamma(ndata+exp(logα)) + lgamma(exp(logα))
end

function likelihood_cluster{T<:AbstractCluster}(vc::Vector{T}, logα::Float64, hp::HyperParameter)
    s = length(vc) * logα
    N = 0
    for c in vc
        N += c.n
        s += lgamma(c.n)
        s += _b(c,hp)
        s -= _b(hp)
    end
    α = exp(logα)
    # s += lgamma(α) # evaluated in like_data
    # s -= lgamma(N+α)
    # println("# data = ", N)
    return s
end

#=
### Evaluation
begin
    ## 1. Collect labels and calc vi
    for wkid in 1:M
        label[indexrange[wkid]] = @fetchfrom wkid+1 send_label_shallow(fetch(fwks[wkid]))
    end
    if iter > 1
        empty!(maptable)
        for (sid,sc) in c.supercc, subid in sc.ids
            maptable[subid] = sid
        end
        for i in 1:ndata
            label[i] = maptable[label[i]]
        end
    end
    vi[iter] = Clustering.varinfo(maximum(gt),gt,maximum(label),label)
    println(" vi = ", vi[iter])

    ## 2. Collect models and calc likelihood
    if iter == 1
        vc = Vector{Cluster}()
        for wkid in 1:M
            cc = @fetchfrom wkid+1 send_cluster(fetch(fwks[wkid]))
            for dc in values(cc)
                push!(vc,dc)
            end
        end
        lll[iter] = like_data + likelihood_cluster(vc, logα, hp)
    else
        lll[iter] = like_data + likelihood_cluster(send_cluster_shallow(c), logα, hp)
    end
    println(" loglikelihood = ", lll[iter])
end
=#
