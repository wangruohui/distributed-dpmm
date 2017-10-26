using PyPlot
using StatsBase
using HDF5

function gen_synthetic(seed::Int)
    srand(seed)
    sigma = 1
    sigma_0 = 10

    K = 50
    center = randn(2,K)*sigma_0

    sizes = rand([1000,2000,5000],K)
    N = sum(sizes)
    println(sizes)
    println(N)

    endpoint = cumsum(sizes)
    startpoint = 1 + vcat(0,endpoint)

    data = randn(2,N)

    quadrant = Array(Int,N)
    for i in 1:N
        if data[1,i]>0
            if data[2,i]>0
                quadrant[i] = 1
            else
                quadrant[i] = 4
            end
        else
            if data[2,i]>0
                quadrant[i] = 2
            else
                quadrant[i] = 3
            end
        end
    end

    label = Array(Int, N)
    for k in 1:K
        data[:,startpoint[k]:endpoint[k]] .+= center[:,k]
        label[startpoint[k]:endpoint[k]] .= k
    end

    index = randperm(N)

    label = label[index]
    quadrant = quadrant[index]
    data = data[:,index]

    idx = rand(1:N,3000)
    close("all")
    figure(figsize=(5,5))
    scatter(data[1,idx],data[2,idx],label="samples")
    # title("Visulization of the Synthetic dataset")
    xlim([-30,40])
    ylim([-40,30])
    legend(loc=4)


    figure(figsize=(5,5))
    scatter(center[1,sizes.==1000],center[2,sizes.==1000];color="b",marker=".",label="size=1K")
    scatter(center[1,sizes.==2000],center[2,sizes.==2000];color="r",marker="*",label="size=2K")
    scatter(center[1,sizes.==5000],center[2,sizes.==5000];color="k",marker="+",label="size=5K")
    legend(loc=4)

    for k in 1:K
        color = "k"
        if sizes[k] == 1000
            color = "b"
        elseif sizes[k] == 2000
            color = "r"
        end
        plot(center[1,k].+2*cos(0:0.01:6.3),center[2,k].+2*sin(0:0.01:6.3);color= color)
        #text(center[1,k],center[2,k],string(sizes[k]))
    end
    # title("Centers and sizes of the Synthetic dataset")
    xlim([-30,40])
    ylim([-40,30])


    filename = "/home/rhwang/DATA/synthetic_$K.h5"
    # ispath(filename) && rm(filename)
    # h5write(filename, "feature", data)
    # h5write(filename, "label", label)
end
