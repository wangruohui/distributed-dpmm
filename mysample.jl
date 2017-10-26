
# High performance sample for CRP and Progressive Merge
function mysample(w::Vector)
    t = rand() * sum(w)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

function mysample(a::AbstractArray, w::Vector)
    i = mysample(w)
    return a[i]
end
