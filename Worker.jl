
type Worker
    data    ::  AbstractMatrix
    label   ::  Vector{Int}
    dcc     ::  DataClusterCollection
    occ     ::  ClusterCollection
    nextid  ::  Int
    idstep  ::  Int
    prob    ::  Vector{Float64} # buffer for performance
    ids     ::  Vector{Int64}   # buffer for performance
end

## Constructor
# Empty for test
Worker(nextid::Int, idstep::Int) = Worker(Matrix{Float64}(),Vector{Int}(),DataClusterCollection(),ClusterCollection(),nextid,idstep,Vector{Int}(),Vector{Int}())

# Randomly initialization with KINIT components
function Worker(data::AbstractMatrix, idstart::Int, idstep::Int, hp::HyperParameter, KINIT::Int=1)
    assert(size(data,1)==hp.dd)
    label = rand((0:KINIT-1)*idstep+idstart,size(data,2))
    dcc = DataClusterCollection()
    for i in eachindex(label)
        add_data!(dcc, label[i], data[:,i], i, hp)
    end
    occ = ClusterCollection()
    nextid = idstart + idstep*KINIT
    prob = Vector{Float64}()
    ids = Vector{Int}()
    return Worker(data, label, dcc, occ, nextid, idstep, prob, ids)
end


# Init using given label assignment
function Worker(data::AbstractMatrix, idstart::Int, idstep::Int, hp::HyperParameter, label::Vector{Int})
    assert(size(data,2)==length(label))
    assert(size(data,1)==hp.dd)
    labelscale = (label-minimum(label))*idstep + idstart
    dcc = DataClusterCollection()
    for i in eachindex(labelscale)
        add_data!(dcc, labelscale[i], data[:,i], i, hp)
    end
    occ = ClusterCollection()
    nextid = maximum(labelscale) + idstep
    prob = Vector{Float64}()
    ids = Vector{Int}()
    return Worker(data, labelscale, dcc, occ, nextid, idstep, prob, ids)
end


## Summary Cluster Information
function report_cluster_size(w::Worker)
    count = Array{Int,1}()
    for id in keys(w.dcc)
        push!(count,w.dcc[id].n)
    end
    println(" # of Clusters = ", length(count), ", Size = ", sort(count))
    println(" Summation = $(sum(count))")
    return count
end


## Local iteration
function local_iteration!(w::Worker,scc::Vector{SuperCluster},hp::HyperParameter,logα::Float64)
    recv_cluster!(w, scc)
    add_occ!(w)
    crp!(w,hp,logα)
    sub_occ!(w)
    return w
end


## CRP
function crp!(w::Worker, hp::HyperParameter, logα::Float64)
    # xi = Array{Float64}(hp.dim)
    for i in eachindex(w.label)
        # println(i)
        # println(length(w.dcc))
        # copy!(xi, 1, w.data, 1+hp.dim*(i-1), hp.dim)
        xi = w.data[:,i]
        zi = w.label[i]

        rm_data!(w.dcc, zi, xi, i, hp)

        calc_prob!(w.prob, w.ids, w.dcc, xi, hp, logα)

        zi = mysample(w.ids, w.prob)
        if zi == -1
            w.label[i] = w.nextid
            w.nextid += w.idstep
        else
            w.label[i] = zi
        end

        add_data!(w.dcc, w.label[i], xi, i, hp)
    end
    return w
end


## Communication
# Send ClusterCollection to Fusion Center
function send_cluster(w::Worker)
    # Make a deepcopy
    cc = ClusterCollection()
    for cid in keys(w.dcc)
        cc[cid] = Cluster(w.dcc[cid])
    end
    return cc
end

# Send label for evaluation
send_label_shallow(w::Worker) = w.label

# Receive SuperClusterCollection from Fusion Center
function recv_cluster!(w::Worker, scc::Vector{SuperCluster})
    # scc should already be a deep copy from Fusion Center and free to be edited
    # 1. Remove ids that does not belong to this worker
    for sc in scc, id in sc.ids
        haskey(w.dcc, id) || setdiff!(sc.ids, id)
    end
    # 2. Merge dc's that belongs to the same super cluster
    # Remember to remove merged clusters from w.dcc
    # Remember to change w.label
    maptable = Dict{Int,Int}()
    for sc in scc
        if length(sc.ids) <= 1
            continue
        end
        firstid::Int = -1006
        for id in sc.ids
            if firstid == -1006
                firstid = id
                assert(id!=-1006)
            else
                setdiff!(sc.ids, id)
                assert(firstid!=id)
                merge_dc!(w.dcc[firstid], w.dcc[id])
                delete!(w.dcc, id)
                assert(!haskey(maptable,id))
                maptable[id] = firstid
            end
        end
        assert(length(sc.ids)<=1)
    end
    # Change w.label
    for i in eachindex(w.label)
        haskey(maptable,w.label[i]) && (w.label[i] = maptable[w.label[i]])
    end
    # 3. Create occ, minus statistics of dc
    occ = ClusterCollection()
    for sc in scc
        assert(length(sc.ids)<=1)
        c = Cluster(sc.n, sc.ss)    # shallow copy
        id::Int = -1022
        if length(sc.ids) == 1
            id = first(sc.ids)
            sub_stat!(c, w.dcc[id])
            c.n == 0 && continue     # do not create if empty
        else
            id = w.nextid
            w.nextid += w.idstep
        end
        assert(!haskey(occ,id))
        occ[id] = c
    end
    #
    w.occ = occ
    return w
end


## Add / Remove "other"
function add_occ!(w::Worker)
    for id in keys(w.occ)
        if haskey(w.dcc,id)
            add_stat!(w.dcc[id], w.occ[id])
        else
            w.dcc[id] = DataCluster(w.occ[id])
        end
    end
    return w
end

function sub_occ!(w::Worker)
    for id in keys(w.dcc)
        haskey(w.occ,id) || continue
        if w.dcc[id].n == w.occ[id].n
            assert(sum(abs2,w.dcc[id].ss-w.occ[id].ss) < eps(Float32))
            delete!(w.dcc, id)
        else
            sub_stat!(w.dcc[id], w.occ[id])
        end
    end
    return w
end
