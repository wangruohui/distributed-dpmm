# Scalable Estimation of Dirichlet Process Mixture Models on Distributed Data

This repository contains our implementation of three proposed algorithms as well as compared baselines in the following paper.

[1] Ruohui Wang, Dahua Lin, [Scalable Estimation of Dirichlet Process Mixture Models on Distributed Data](https://www.ijcai.org/proceedings/2017/646), *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence
Main track (IJCAI-17)*. Pages 4632-4639.

Please kindly cite our paper if the code helps you. Thank you.

```latex
@inproceedings{ijcai2017-646,
  author    = {Ruohui Wang, Dahua Lin},
  title     = {Scalable Estimation of Dirichlet Process Mixture Models on Distributed Data},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {4632--4639},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/646},
  url       = {https://doi.org/10.24963/ijcai.2017/646},
}
```

# Usage Guide
* [Install Julia and required packages](#install-julia-and-required-packages)
* [Prepare some data](#prepare-some-data)
* [Test the CGS baseline](#test-the-cgs-baseline)
* [Test the CGS baseline for multiple times](#test-the-cgs-baseline-for-multiple-times)
* [Test three proposed algorithms](#test-three-proposed-algorithms)
* [Implementation of other baselines](#implementation-of-other-baselines)
* [Scale to multiple servers](#scale-to-multiple-servers)

## Install Julia and required packages

Our program is implemented using the [Julia programming language](https://julialang.org/).
It can be easily scaled to multiple processors and multiple servers with Julia's built-in support for [parallel computing](https://docs.julialang.org/en/latest/manual/parallel-computing).
So in order to run our program, you need to first install the Julia programming language, following [this link](https://julialang.org/downloads/).
Once installed, launch Julia from the terminal by executing `julia` and install required packages using the following commands (within Julia).
```julia
julia> Pkg.add("JLD")
julia> Pkg.add("HDF5")
julia> Pkg.add("PyPlot")
julia> Pkg.add("StatsBase")
julia> Pkg.add("Clustering")
julia> Pkg.add("Distributions")
julia> Pkg.add("Hungarian")
```

## Prepare some data

The program need to be run with data.
We used three datasets in our paper [1], a synthetic one, one derived from [ImageNet](http://www.image-net.org/), and one derived from [NYT Corpus](https://catalog.ldc.upenn.edu/ldc2008t19).  

The Synthetic one can be generated using code [`gen_synthetic.jl`](gen_synthetic.jl).
Just make sure to uncomment line 84-86 and set a correct saving path before running the code.
In Julia, execute
```julia
julia> include("gen_synthetic.jl")
julia> gen_synthetic(101)
```
 `101` is the random seed that generates the data used in the paper. You can change to other seeds.

 The prepared ImageNet and NYT corpus dataset (as well as the synthetic one generated from above codes) can be downloaded from [here](https://www.dropbox.com/sh/b9d0tpmnzq2ky11/AABja2sT1Ap-vCBpctLa-JD5a?dl=0).
 The entire prepared NYT dataset is too large to be uploaded. So we provide two subsets (articles in January 1987 and articles in whole year 1987) only.
 Download them to your local storage and modify paths in [`ExperimentSetting.jl`](ExperimentSetting.jl) accordingly.

You can also use your own dataset.
The program takes a matrix with all feature vectors and a vector of all labels.
Remember to add an entry in [`ExperimentSetting.jl`](ExperimentSetting.jl).

## Test the CGS baseline

0. Mare sure the path is correctly set in [`ExperimentSetting.jl`](ExperimentSetting.jl).
1. Launch Julia from the terminal
```bash
$ julia
```
2. Include the code
```julia
julia> include("evalCRP.jl")
```
3. Run with the synthetic dataset.
```julia
evalCRP(10, 50, :s, true)
```
The above code will initialize with 10 randomly assigned labels and perform collapsed Gibbs sampling for 50 iterations on the dataset.
`:s` stands for the synthetic dataset.
Symbols for other datasets can be found in [`ExperimentSetting.jl`](ExperimentSetting.jl).

The program will output summary of the result every iteration like:
```
Iter = 50
** This is the 50-th iteration **

 runtime = 1.602912625
 ** This iteration takes these seconds to run **

 # of Clusters = 60, Size = [1, 3, 8, 12, 87, 177, 310, 348, 367, 452, 460, 542, 595, 793, 917, 967, 983, 997, 999, 1003, 1004, 1005, 1008, 1014, 1080, 1774, 1912, 1912, 1913, 1983, 1984, 1987, 1995, 2000, 2004, 2011, 2020, 2022, 2029, 2069, 2090, 3358, 4235, 4710, 4968, 4988, 4990, 4995, 4998, 5000, 5000, 5000, 5002, 5002, 5012, 5058, 5059, 5080, 5223, 6485]
 ** There are totally 60 clusters proposed, their sizes (number of samples contained) are [1, 3 ... 6485] respectively, sorted ascendingly **

 Summation = 141000    ** There are totally 141000 samples. This is for verification purpose only. Sometimes programming errors will lead to loss of samples. **

 loglikelihood = -929192.2782973386
 ** the Likelihood of the model **

 vi = 0.7586535041975777
 ** Variational Information criteria (see paper [1] for detail) **

 count = 60.0
 ** Total number of proposed components, same as above **
```

This is the desired result.

## Test the CGS baseline for multiple times

You may need to run the baseline for multiple times and average the results.
To do so, please follow below procedures.  

0. Mare sure the path is correctly set in [`ExperimentSetting.jl`](ExperimentSetting.jl).
1. Launch Julia with multiple workers (5 workers here). Each worker will reside on one CPU core.
```bash
$ julia -p 5
```
2. Include the code
```julia
julia> include("evalCRP.jl")
```
3. Run 4 parallel CGS baseline on the synthetic dataset.
```julia
evalCRPrepeat(4, 50, :s)
```
The above code will run the CGS baseline 4 times, each for 50 iterations, in parallel, and output the averaged results.

Note: in order to give an accurate measure on running time, the number of parallel experiments should be less than the number of CPU cores (with hyper threading disabled) on the computer.

## Test three proposed algorithms

Codes for testing our proposed methods are provided in [`evalSync.jl`](evalSync.jl) and [`evalAsync.jl`](evalAsync.jl).
Similar to [`evalCRP.jl`](evalCRP.jl), a function called `evalXXX` will evaluate the algorithm once and a function called `evalXXXrepeat` will evaluate the algorithm for multiple times and report averaged results.
But unlike the serial Gibbs sampler, our proposed methods are implemented in a parallel manner.
As a result, `evalXXX` itself requires launching multiple Julia workers.
However, `evalXXXrepeat` function does not require more workers.
It will simply repeat `evalXXX` for multiple times and average the results.

For example, there are 20 physical cores on my computer.
1. Launch Julia with 21 workers.
```bash
$ julia -p 21
```
2. Evaluate the algorithm (on the synthetic dataset)
- Progressive consolidation
```julia
julia> include("evalSync.jl"); evalSyncDistribute(20, 50, 0, :s)
```
The above code will run progressive consolidation using 20 workers and for 50 iterations. The third argument should be kept as 0.  
- Pooled consolidation
```julia
julia> include("evalSync.jl"); evalSyncDistribute(20, 50, 100, :s)
```
The above code will run pooled consolidation using 20 workers and for 50 iterations. The third argument indicates the number of MCMC steps performed in the fusion center. See section 4.2.2 in the paper [1] for detail.
- Asynchronous algorithm
```julia
julia> include("evalAsync.jl"); evalAsyncDistribute(20, 50, 100, :s)
```
The arguments take the same meaning as they do in pooled consolidation.

3. Evaluate the algorithm for multiple times (3 times in below examples)
- Progressive consolidation
```julia
julia> include("evalSync.jl"); evalSyncRepeat(3, 20, 50, 0, :s)
```
- Pooled consolidation
```julia
julia> include("evalSync.jl"); evalSyncRepeat(3, 20, 50, 100, :s)
```
- Asynchronous algorithm
```julia
julia> include("evalAsync.jl"); evalAsyncRepeat(3, 20, 50, 100, :s)
```
Note: In these implementation, data distributed to the same worker will be initialized using the same label, which is the id of the worker.

## Implementation of other baselines

We also implemented other baselines, including AVparallel [2] (a improved version), SubC [3], SliceMR [4] and a modification of our proposed method using the Hungarian merging policy proposed in [5], all in Julia.

[2] Sinead Williamson, Avinava Dubey, and Eric Xing. **Parallel markov chain monte carlo for nonparametric mixture models**. *In Proceedings of the 30th International Conference on Machine Learning*, pages 98–106, 2013.

[3] Jason Chang and John W Fisher III. **Parallel sampling of dp mixture models using sub-clustersplits**. *In Advances in Neural Information Processing Systems*, pages 620–628, 2013.

[4] Hong Ge, Yutian Chen, Moquan Wan, and Zoubin Ghahramani. **Distributed inference for dirichlet process mixture models**. *In Proceedings of the 32nd International Conference on Machine Learning (ICML-15)*, pages 2276–2284, 2015.

[5] Trevor Campbell, Julian Straub, John W Fisher III, and Jonathan P How. **Streaming, distributed variational inference for bayesian nonparametrics**. *In Advances in Neural Information Processing Systems*, pages 280–288, 2015.

These baselines are also implemented in a parallel manner. So in order to evaluate them, Julia need to be launched with multiple workers, e.g. `julia -p 21`.

Detailed instructions are list as follows.

1. AVparallel [2]

Navigate to the root directory and launch Julia with multiple worker, then execute
```julia
julia> include("evalAV.jl"); evalAVparallel(20, 50, 100, 1, :s)
```
Here: `20` is the number of workers. `50` is the total number of iterations. `100` is the number of MCMC steps. `1` is the number of initial components (on each worker). `:s` stands for the synthetic dataset.

2. SubC [3]

Navigate to `SubC` directory and launch Julia with multiple workers, then execute
```julia
julia> include("SubcDistribute.jl"); evalSubcDistribute(20, 50, 100, 1, :s)
```
Arguments are of the same meaning as above.

3. SliceMR [4]

Navigate to `SliceMR` directory and launch Julia with multiple workers, then execute
```julia
julia> include("evalSliceDistributed.jl"); evalSliceMRDistributed(20, 50, 1, :s)
```
Here: `20` is the number of workers. `50` is the total number of iterations. `1` is the number of initial components (on each worker). `:s` stands for the synthetic dataset.

4. Hungarian [5]

Navigate to the root directory and launch Julia with multiple worker, then execute
```julia
julia> include("evalSync.jl"); evalSyncDistribute(20, 50, -1, :s)
```
Here: `20` is the number of workers. `50` is the total number of iterations. The third argument `-1` indicates the program to perform Hungarian merging policy. `:s` stands for the synthetic dataset.

## Scale to multiple servers

With Julia's native support for parallel computing, it is easy to scale our program to multiple servers without modifying the code.
All we need to do is to set up a distributed environment with multiple servers for Julia.
Since we implemented both our proposed methods and other baselines using Julia.
All of them can be evaluated under multi-server setting.

In the following instructions, I use two servers with IP `10.1.72.21` and `10.1.72.22` as example.
It is possible to use more servers and the command is similar.

### Install Julia
Install the Julia programming language on all servers and set `$PATH` environment variables correctly.
Make sure that, on each server, by typing `julia` from the console, Julia can be launched correctly.

### Configure password-less SSH
Create user accounts on all servers (with the same username for convenience) and configure password-less SSH login.
There is a bunch of tutorials available on the web, here is an [example](https://www.tecmint.com/ssh-passwordless-login-using-ssh-keygen-in-5-easy-steps/).

After this step, you should be able to login between any two servers and in both direction.
Note that for the first time to login, the server will prompt for key fingerprint verification.
You need to manually accept it before running the Julia client.
Otherwise it will block the Julia client connecting between servers.
This action need to be performed once between any directional pair of servers.

Just make sure from either server, after typing `ssh $ANOTHER_IP_ADDRESS`, the console of the destination server appears without any interruption.

### Launch Julia workers from multiple nodes
Launch Julia from one server (`10.1.72.21` here as example) by typing `julia` from the console.
After it starts, issue
```julia
julia> addprocs([("10.1.72.21",15),("10.1.72.22",15)])
```

This command will start 15 workers on both servers.
There will be 30 workers in total.

If success, this function will return a list of worker ids like below.
```julia
julia> addprocs([("10.1.72.21",15),("10.1.72.22",15)])
30-element Array{Int64,1}:
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
```

### Include the code and evaluate it
Then everything is like before.
For example, we can issue the following command to evaluate the progressive consolidation using 30 workers
```julia
julia> include("evalSync.jl"); evalSyncRepeat(3, 30, 50, 0, :s)
```
Note that we modified the second argument from 20 to 30.

The initialization step could be rather slow from a single minute (For synthetic dataset) to tens of minutes (for the entire NYT corpus), due to code distribution, code compilation and data distribution. Just be patient.
