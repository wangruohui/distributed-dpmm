
@everywhere global NOASSERT = true

function get_exp_setting(dataset::Symbol)
    feature = 0
    label = 0
    logα = 0
    hp = 0
    if dataset == :dummydoc
        logα = 0.0
        hp = MultinomialDir(ones(Float64,1000))
        feature = sparse(rand(1:1000,10000),rand(1:500,10000),rand(1:50,10000))
        label = rand(1:10,500)
    elseif dataset == :nyt1m
        logα = 0.0
        feature = load(homedir()*"/DATA/nyt-1987-1-feat.jld","features")
        label = load(homedir()*"/DATA/nyt-1987-1-label.jld","labels")
        hp = MultinomialDir(ones(Float64,size(feature,1)))
    elseif dataset == :nyt1y
        logα = 0.0
        feature = load(homedir()*"/DATA/nyt-1987-feat.jld","features")
        label = load(homedir()*"/DATA/nyt-1987-label.jld","labels")
        hp = MultinomialDir(ones(Float64,size(feature,1)))
    elseif dataset == :nytall
        logα = 0.0
        feature = load(homedir()*"/DATA/nyt-feat-fil.jld","features")
        label = load(homedir()*"/DATA/nyt-label-fil.jld","labels")
        hp = MultinomialDir(ones(Float64,size(feature,1)))
    elseif dataset == :s
        filename = homedir() * "/DATA/synthetic.h5"
        feature = HDF5.h5read(filename, "feature")
        label = HDF5.h5read(filename, "label")
        logα = 0.0
        hp = IsotropicGaussian(size(feature,1),10,1)
    elseif dataset == :i48
        # 48-dim
        # sigma of center : 6.885933753345786
        # sigma of all : 9.21927155012262
        # sigma within class : 6.127468700166469
        filename = homedir() * "/DATA/imagenet_xczhang_48.h5"
        feature = HDF5.h5read(filename, "feature")
        label = HDF5.h5read(filename, "label")
        logα = 0.0
        hp = IsotropicGaussian(size(feature,1),8,8)
    else
        error("unknown profile")
        assert(false)
    end

    assert(size(feature, 2) == length(label))
    println("Size of Data: ", size(feature))
    println("Parameter, logα = ", logα)
    isa(hp,IsotropicGaussian) && println("Parameter, hp = ", hp)

    return feature,label,logα,hp
end
