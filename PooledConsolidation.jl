
function pooled_consolidation(c::Center, T::Int, logα::Float64, hp::HyperParameter)
    pm = [0,0]
    am = [0,0]
    ps = [0,0]
    as = [0,0]
    uc = [0,0]

    supercc = c.supercc
    subcc = c.subcc
    mergetb = c.mergetb
    splittb = c.splittb
    cm = Cluster(hp)

    for t in 1:T
        ph = 2*t<T ? 1 : 2
        if rand() > 0.5
            #### Propose Merge
            isempty(mergetb) && continue
            pm[ph] += 1
            #- M1. Sample MergeTable, get (A,B) and P1 = P(x->x')
            (id1,id2), LP1 = sample_table(mergetb)
            #- M2. Temporarily
            K = c.nextid
            #- M2.1. add (A+B) to supers (A,B are not removed)
            supercc[K] = merge_sc(supercc[id1], supercc[id2])
            #- M2.2. modify SplitTable +(A+B) (A,B are not removed)
            add_table!(splittb, supercc, subcc, K, logα, hp)
            #- M2.3. sample SplitTable, but skip A and B. Return P3 = select(A+B)
            LP3 = log_sample_prob(splittb, K, id1, id2)
            #- M3. Calc P4 = split (A+B) to (A,B)
            LP4 = log_split_prob(supercc[K], supercc[id1].ids, subcc, logα, hp)
            #- M4. Calc P2 = P(x')/P(x)
            LP2 = logρ(supercc[id1],supercc[id2],supercc[K],logα,hp)
            #- M5. Calc accept ratio
            """
            println()
            println((id1,id2))
            println(supercc[id1])
            println(supercc[id2])
            println(LP1)
            println(LP2)
            println(LP3)
            println(LP4)
            println()
            """
            assert(LP1 > -Inf)
            assert(LP2 > -Inf)
            assert(LP3 > -Inf)
            assert(LP4 > -Inf)
            accept_thre = min(1, exp(LP2 - LP1 + LP3 + LP4))
            # println(' ', exp(LP2 - LP1 + LP3 + LP4))
            if rand() < accept_thre
                #### Accept Merge
                am[ph] += 1
                #- MA1. modify supercc (-A,-B), (A+B) is already added
                delete!(supercc, id1)
                delete!(supercc, id2)
                #- MA2. modify subcc
                for subid in supercc[K].ids
                    assert(subcc[subid].super == id1 || subcc[subid].super == id2)
                    subcc[subid].super = K
                end
                #- MA3. modify MergeTable (-A,-B,+(A+B))
                delete_table!(mergetb, id1)
                delete_table!(mergetb, id2)
                add_table!(mergetb, supercc, K, logα, hp)
                ## MA4. modify SplitTable (-A,-B), (A+B) is already added
                delete_table!(splittb, id1)
                delete_table!(splittb, id2)
                #- MA5. modify c.nextid
                c.nextid += 1
            else
                #### Reject Merge
                ## MR1. recover supercc
                delete!(supercc, K)
                ## MR2. recover SplitTable
                delete_table!(splittb, K)
            end
        else
            #### Propose Split
            isempty(splittb) && continue
            ps[ph] += 1
            #- S1. Sample SplitTable, get A and P1 = P(select A)
            sid, LP1 = sample_table(splittb)
            #- S2. Try split A to B+C, and return P2 = P(split A to B+C)
            isuc, LP2, split_clusters = propose_split(supercc[sid], subcc, logα, hp)
            if isuc
                #- S*. remain current status, always accept
                uc[ph] += 1
            else
                #- S3. a real split proposal, Temporarily
                K1 = c.nextid
                K2 = c.nextid+1
                #- S3.1. add B,C to supers (A is not removed)
                supercc[K1] = split_clusters[1]
                supercc[K2] = split_clusters[2]
                #- S3.2. modify MergeTable (+B,+C) (A is not removed)
                add_table!(mergetb, supercc, K1, logα, hp)
                add_table!(mergetb, supercc, K2, logα, hp)
                #- S3.3. return P3 = select(B,C) = P(x'->x)
                LP3 = log_sample_prob(mergetb, (K1, K2), sid)
                #- S4. Calc P4 = P(x')/P(x)
                LP4 = -logρ(supercc[K1],supercc[K2],supercc[sid],logα,hp)
                #- S5. Calc accept ratio
                assert(LP1 > -Inf)
                assert(LP2 > -Inf)
                assert(LP3 > -Inf)
                assert(LP4 > -Inf)
                accept_thre = min(1, exp(LP4 + LP3 - LP1 - LP2))
                if rand() < accept_thre
                    #### Accept Split
                    as[ph] += 1
                    #- SA1. modify supercc -A, (B,C) is already added
                    delete!(supercc, sid)
                    #- SA2. modify subcc
                    for KK in [K1,K2], subid in supercc[KK].ids
                        assert(subcc[subid].super==sid)
                        subcc[subid].super = KK
                    end
                    #- SA3. modify MergeTable -A, (B,C) is already added
                    delete_table!(mergetb, sid)
                    #- SA4. modify SplitTable (-A,+B,+C)
                    delete_table!(splittb, sid)
                    add_table!(splittb, supercc, subcc, K1, logα, hp)
                    add_table!(splittb, supercc, subcc, K2, logα, hp)
                    #- SA5. modify modify c.nextid
                    c.nextid += 2
                else
                    #### Reject Split
                    ## SR1. recover supercc
                    delete!(supercc, K1)
                    delete!(supercc, K2)
                    ## SR2. recover MergeTable
                    delete_table!(mergetb, K1)
                    delete_table!(mergetb, K2)
                end
            end
        end
    end
    println("  # Propose Merge = ", pm)
    println("  # Accept Merge = ", am)
    println("  # Propose Split = ", ps)
    println("  # Unchange = ", uc)
    println("  # Accept Split = ", as)

    return c
end
