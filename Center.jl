
const MergeTable = Dict{Tuple{Int,Int},Float64}
const SplitTable = Dict{Int,Float64}
const McmcTable = Union{MergeTable,SplitTable}

type Center
    subids   :: Vector{Set{Int}}        # wkid -> subids submitted from this worker
    subcc    :: SubClusterCollection
    supercc  :: SuperClusterCollection
    mergetb  :: MergeTable
    splittb  :: SplitTable
    nextid   :: Int                     # id of supercluster
end

include("McmcTable.jl")
include("McmcSplit.jl")
include("PooledConsolidation.jl")

## Constructor
function Center(nwk::Int)
    subids = Array{Set{Int}}(nwk)
    for i in 1:nwk
        subids[i] = Set{Int}()
    end
    subcc = SubClusterCollection()
    supercc = SuperClusterCollection()
    mergetb = MergeTable()
    splittb = SplitTable()
    return Center(subids, subcc, supercc, mergetb, splittb, 1)
end


## Summary Cluster Information
function report_cluster_size(c::Center)
    count = Array{Int}(0)
    for id in keys(c.supercc)
        push!(count,c.supercc[id].n)
    end
    println()
    println(" Center Clusters = ", length(count), ", Size = ", sort(count))
    println(" Summation = $(sum(count))")
    return c
end


## Send SuperClusterCollection to worker
# send_cluster(c::Center) = deepcopy(c.supercc)


## Send a shallow copy of SuperClusterCollection to worker
# NEED deepcopy in single core version
function send_cluster_shallow(c::Center)
    vc = Vector{SuperCluster}()
    for sc in values(c.supercc)
        push!(vc, sc)
    end
    return vc
end

function send_cluster_shallow(c::Center, wkid::Int)
    vc = Vector{SuperCluster}()
    for sc in values(c.supercc)
        sccopy = SuperCluster(Set{Int}(),sc.n,sc.ss)
        for id in c.subids[wkid]
            # println(id)
            # println(sc.ids)
            in(id,sc.ids) && union!(sccopy.ids,id)
            # println(sccopy.ids)
        end
        push!(vc, sccopy)
    end
    return vc
end


## Receive ClusterCollection from worker
## -> Progressive Consolidation <-
recv_cluster_prog!(c::Center, wkid::Int, cc::ClusterCollection, logα::Float64, hp::HyperParameter) = recv_cluster!(c, wkid, cc, logα, hp, :prog)
## -> MCMC Pooled Consolidation <-
recv_cluster_pool!(c::Center, wkid::Int, cc::ClusterCollection) = recv_cluster!(c, wkid, cc, 0.0, IsotropicGaussian(0,0,0), :mcmc)
## -> Hungarian Algorithm for Consolidation <-
recv_cluster_hung!(c::Center, wkid::Int, cc::ClusterCollection, logα::Float64, hp::HyperParameter) = recv_cluster!(c, wkid, cc, logα, hp, :hung)

function recv_cluster!(c::Center, wkid::Int, cc::ClusterCollection, logα::Float64, hp::HyperParameter, merge_algo::Symbol)

    ## !!ATTENTION!! cc should already be a deepcopy
    subids = c.subids[wkid]
    subcc = c.subcc
    supercc = c.supercc

    #### STEP 1: update existing subclusters in center based on cc
    for subid in subids
        sub2update = subcc[subid]
        superid = sub2update.super
        super2update = supercc[superid]

        #### 1 remove this sub from its super
        rm_sub!(super2update, sub2update, subid)

        #### 2 Check from uploaded ClusterCollection
        if haskey(cc,subid)
            #### 2a.1 update information (shalow copy)
            sub2update.n = cc[subid].n
            sub2update.ss = cc[subid].ss
            #### 2a.2 add back to super
            add_sub!(super2update, sub2update, subid)
        else
            #### 2b.1 remove this subid from c.subids
            setdiff!(subids, subid)
            #### 2b.2 remove this super if its empty
            if isempty(super2update.ids)
                assert(super2update.n==0)
                assert(sum(abs2,super2update.ss) < eps(Float32))
                delete!(supercc, superid)
            end
        end
        #### 3 Remove this from CC, useless for future
        delete!(cc,subid)
    end

    if isempty(cc) == 0
        ## There is no new proposed clusters in cc
        return c
    end

    #### STEP 2: merge new created clusters in cc to center
    prob = Vector{Float64}()
    spids = Vector{Int}()
    cm = Cluster(hp)

    # Hungarian method preparation
    mergetable = 0
    if merge_algo == :hung
        R = zeros(Float64, length(cc),length(cc)+length(supercc))
        taskids = collect(keys(cc))
        agentids = collect(keys(supercc))
        for (tid, ccid) in enumerate(taskids), (aid, suid) in enumerate(agentids)
            c1 = cc[ccid]
            c2 = supercc[suid]
            cm = Cluster(c1)
            add_stat!(cm, c2)
            R[tid, aid] += lgamma(c1.n) + _b(c1,hp) - _b(hp)
            R[tid, aid] += lgamma(c2.n) + _b(c2,hp) - _b(hp)
            R[tid, aid] -= lgamma(cm.n) + _b(cm,hp) - _b(hp)
        end
        assignment, _ = hungarian(R)
        mergetable = Dict{Int,Int}()
        for (i,a) in enumerate(assignment)
            if a <= length(supercc)
                mergetable[taskids[i]] = agentids[a]
            else
                mergetable[taskids[i]] = -1
            end
        end
    end

    for id in keys(cc)
        #### A1: assign superid = -1 for MCMC pooled consolidation
        spid = -1
        #### B1: assign sample superid based on merge_split_ratio for progressive consolidation
        if merge_algo == :prog
            empty!(prob)
            empty!(spids)
            for spid in eachindex(supercc)
                push!(prob, logρ!(supercc[spid], cc[id], cm, logα, hp))
                push!(spids, spid)
            end
            push!(prob, 0)
            push!(spids, -1)

            maxlogprob = maximum(prob)
            @simd for i in eachindex(prob)
                prob[i] = exp(prob[i] - maxlogprob)
            end

            spid = mysample(spids,prob)
        elseif merge_algo == :hung
            spid = mergetable[id]
        end

        if spid != -1
            #### B2, merge to exising supercluster
            # Create SubCluster
            assert(!haskey(subcc,id))
            subcc[id] = SubCluster(spid, cc[id].n, cc[id].ss) # shallow copy
            # Add to SuperCluster
            add_sub!(supercc[spid], subcc[id], id)
        else
            #### A2/B3, create singleton supercluster
            new_superid = c.nextid
            c.nextid += 1
            # Create SubCluster
            assert(!haskey(subcc,id))
            subcc[id] = SubCluster(new_superid, cc[id].n, cc[id].ss) # shallow copy
            # Create singleton SuperCluster
            assert(!haskey(supercc,new_superid))
            supercc[new_superid] = SuperCluster(Set{Int}(id), cc[id].n, deepcopy(cc[id].ss))
        end
        ##!! important don't forget!
        assert(!in(subids,id))
        union!(subids,id)
        delete!(cc,id)
    end

    assert(isempty(cc))
    return c
end
