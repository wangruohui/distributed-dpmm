
global q2 = 1
global q3 = 1

## Calculate Probability Table
# 1. Erase and Re-calculate ALL
function calc_table!(c::Center, logα::Float64, hp::HyperParameter)
    mt = c.mergetb
    st = c.splittb
    supercc = c.supercc
    subcc = c.subcc

    empty!(mt)
    empty!(st)

    tmpcm = Cluster(hp)
    for id1 in keys(supercc), id2 in keys(supercc)
        id1 > id2  &&  (mt[id1,id2] = logρ!(supercc[id1], supercc[id2], tmpcm, logα, hp) * q2)
    end

    for id in keys(supercc)
        superc = supercc[id]
        length(superc.ids) > 1  &&  (st[id] = -log_no_split_prob(superc,subcc,logα,hp) * q3)
    end

    return c
end

# 2 TODO Erase and Re-calculate based on worker id


## Add item to table
function add_table!(mt::MergeTable, supercc::SuperClusterCollection, id::Int, logα::Float64, hp::HyperParameter)
    tmpcm = Cluster(hp)
    for id2 in keys(supercc)
        mt[id,id2] = logρ!(supercc[id], supercc[id2], tmpcm, logα, hp) * q2
    end
    delete!(mt,(id,id))
end

function add_table!(st::SplitTable, supercc::SuperClusterCollection, subcc::SubClusterCollection, id::Int, logα::Float64, hp::HyperParameter)
    superc = supercc[id]
    length(superc.ids) > 1  &&  (st[id] = -log_no_split_prob(superc,subcc,logα,hp) * q3)
end


## Delete item from table
function delete_table!(mt::MergeTable, id::Int)
    for (id1,id2) in keys(mt)
        id1==id && delete!(mt,(id1,id2))
        id2==id && delete!(mt,(id1,id2))
    end
end

function delete_table!(st::SplitTable, id::Int)
    # There is possibility that id is unable to be splitted
    # thus not in splittable. An assert is not proper
    delete!(st,id)
end


## Sample from table
function sample_table(mt::McmcTable)
    max_log_prob = maximum(values(mt))
    sum_exp_prob = 0.0
    for log_prob in values(mt)
        sum_exp_prob += exp(log_prob - max_log_prob)
    end
    thres = rand()*sum_exp_prob
    acc = 0.0
    for id in keys(mt)
        this_prob = exp(mt[id] - max_log_prob)
        next_acc = acc + this_prob
        if next_acc > thres
            return id, mt[id] - max_log_prob - log(sum_exp_prob)
        else
            acc = next_acc
        end
    end
    assert(false)
end

## Calc probability from McmcTable when calc P(x'->x)
# 1. When propose split, x'=(A,B), x=(A+B)
function log_sample_prob(mt::MergeTable, tarid::Tuple{Int,Int}, skip1::Int)
    assert(tarid[1]!=skip1)
    assert(tarid[2]!=skip1)

    max_log_prob = -Inf
    for pair in keys(mt)
        pair[1]==skip1 && continue
        pair[2]==skip1 && continue
        max_log_prob = max(max_log_prob, mt[pair])
    end
    assert(max_log_prob>-Inf)

    sum_exp_prob = 0.0
    for pair in keys(mt)
        pair[1]==skip1 && continue
        pair[2]==skip1 && continue
        log_prob = mt[pair]
        sum_exp_prob += exp(log_prob - max_log_prob)
    end

    return mt[tarid] - max_log_prob - log(sum_exp_prob)
    # for id in keys(mt)
    #     if id == tarid
    #         return mt[id] - max_log_prob - log(sum_exp_prob)
    #     end
    # end
    # assert(false)
end

# 2. When propose merge, x'=A+B, x=(A,B)
function log_sample_prob(st::SplitTable, tarid::Int, skip1::Int, skip2::Int)
    assert(tarid!=skip1)
    assert(tarid!=skip2)

    max_log_prob = -Inf
    for id in keys(st)
        skip1==id && continue
        skip2==id && continue
        max_log_prob = max(max_log_prob, st[id])
    end
    assert(max_log_prob>-Inf)

    sum_exp_prob = 0.0
    for id in keys(st)
        skip1==id && continue
        skip2==id && continue
        log_prob = st[id]
        sum_exp_prob += exp(log_prob - max_log_prob)
    end

    return st[tarid] - max_log_prob - log(sum_exp_prob)
    # for id in keys(st)
    #     if id == tarid
    #         return st[id] - max_log_prob - log(sum_exp_prob)
    #     end
    # end
    # assert(false)
end
