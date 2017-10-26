
abstract type AbstractCluster end

type Cluster <: AbstractCluster
    n   ::  Int             # Number of samples
    ss  ::  Vector{Float64} # Sufficient Statistics
end

type DataCluster <: AbstractCluster
    ids ::  Set{Int}        # IDs of data in the WORKER
    n   ::  Int
    ss  ::  Vector{Float64}
end

type SubCluster <: AbstractCluster
    super :: Int            # Merged to which supercluster
    n     :: Int
    ss    :: Vector{Float64}
end

type SuperCluster <: AbstractCluster
    ids ::  Set{Int}        # IDs of sub clusters
    n   ::  Int
    ss  ::  Vector{Float64}
end

const ClusterCollection = Dict{Int,Cluster}
const DataClusterCollection = Dict{Int,DataCluster}
const SubClusterCollection = Dict{Int, SubCluster}
const SuperClusterCollection = Dict{Int, SuperCluster}


## Constructor
Cluster(hp::HyperParameter) = Cluster(0,zeros(Float64,hp.ds))
Cluster(c::AbstractCluster) = Cluster(c.n,deepcopy(c.ss))

DataCluster(hp::HyperParameter) = DataCluster(Set{Int}(),0,zeros(Float64,hp.ds))
DataCluster(c::AbstractCluster) = DataCluster(Set{Int}(),c.n,deepcopy(c.ss))

SuperCluster(hp::HyperParameter) = SuperCluster(Set{Int}(),0,zeros(Float64, hp.ds))
SuperCluster(c::AbstractCluster,id::Int) = SuperCluster(Set{Int}(id),c.n,deepcopy(c.ss))


## Manipulate Statistics
function add_stat!(c1::AbstractCluster, c2::AbstractCluster)
    assert(length(c1.ss)==length(c2.ss))
    @simd for d in eachindex(c1.ss)
        @inbounds c1.ss[d] += c2.ss[d]
    end
    c1.n += c2.n
end

function sub_stat!(c1::AbstractCluster, c2::AbstractCluster)
    assert(length(c1.ss)==length(c2.ss))
    @simd for d in eachindex(c1.ss)
        @inbounds c1.ss[d] -= c2.ss[d]
    end
    c1.n -= c2.n
    assert(c1.n >= 0)
end

# function copy_stat!(c1::AbstractCluster, c2::AbstractCluster)
#     assert(length(c1.ss)==length(c2.ss))
#     c1.n = c2.n
#     copy!(c1.ss,c2.ss)
# end

# function empty_stat!(ac::AbstractCluster)
#     ac.n = 0
#     fill!(ac.ss,0)
# end


## Data Cluster Basics
function is_empty_cluster(dc::DataCluster)
    if dc.n == 0
        assert(isempty(dc.ids)) ## not true if plused other
        assert(sum(abs2,dc.ss) < eps(Float32))
        return true
    else
        return false
    end
end

function merge_dc!(c1::DataCluster, c2::DataCluster)
    add_stat!(c1, c2)
    NOASSERT || assert(isempty(intersect(c1.ids,c2.ids)))
    union!(c1.ids, c2.ids)
end


## Add / Remove data::Vector in DataCluster
function add_data!(c::DataCluster, x::AbstractVector, id::Int, hp::HyperParameter)
    assert(!in(id,c.ids))
    add_ss!(c.ss, x, hp)    # Type and length check
    c.n += 1
    union!(c.ids, id)
end

function rm_data!(c::DataCluster, x::AbstractVector, id::Int, hp::HyperParameter)
    assert(in(id,c.ids))
    sub_ss!(c.ss, x, hp)    # Type and length check
    c.n -= 1
    setdiff!(c.ids, id)
end

## Add / Remove data::Vector in DataClusterCollection
## TODO seems only CRP use this function, merge to CRP
function add_data!(cc::DataClusterCollection, cid::Int, x::AbstractVector, did::Int, hp::HyperParameter)
    haskey(cc,cid) || (cc[cid] = DataCluster(hp))
    add_data!(cc[cid], x, did, hp)
end

function rm_data!(cc::DataClusterCollection, cid::Int, x::AbstractVector, did::Int, hp::HyperParameter)
    assert(haskey(cc, cid))
    rm_data!(cc[cid], x, did, hp)
    is_empty_cluster(cc[cid]) && delete!(cc, cid)
end


## Add / Remove subcluster in supercluster
## TODO seems only recv_cluster(::Center) use this function
function add_sub!(s::SuperCluster, c::AbstractCluster, cid::Int)
    assert(!in(cid,s.ids))
    add_stat!(s,c)
    union!(s.ids, cid)
end

function rm_sub!(s::SuperCluster, c::AbstractCluster, cid::Int)
    assert(in(cid,s.ids))
    sub_stat!(s,c)
    setdiff!(s.ids, cid)
end


## Merge two Super Cluster
function merge_sc!(sc1::SuperCluster, sc2::SuperCluster)
    add_stat!(sc1,sc2)
    NOASSERT || assert(isempty(intersect(sc1.ids,sc2.ids)))
    union!(sc1.ids,sc2.ids)
end

function merge_sc(sc1::SuperCluster, sc2::SuperCluster)
    sc = deepcopy(sc1)
    merge_sc!(sc,sc2)
    return sc
end
