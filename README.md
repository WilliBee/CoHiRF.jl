# CoHiRF.jl
A Julia implementation of "CoHiRF: A Scalable and Interpretable Clustering Framework for High-Dimensional Data" https://arxiv.org/abs/2502.00380

Adapted in part from https://github.com/BrunoBelucci/cohirf-arxiv/

## Installing Julia

See here https://julialang.org/downloads/#about_juliaup. Install with `juliaup` is the recommended and easiest way.

## Package installation
Launch Julia and create environment (inside the REPL, type `]` on the keyboard to access the Pkg interface):
```
(@v1.12) pkg> activate test_cohirf
```

Then add the package using its GitHub URL (the package is not registered) :
```
(test_cohirf) pkg> add https://github.com/WilliBee/CoHiRF.jl/
```

## Usage

Now back to Julia mode in the REPL by typing backspace, use `run_cohirf_iterations` like this :
```julia
using Random
using CoHiRF
n_features = 14427
n_clusters = 5
samples_per_cluster = 1000
Δ = 100
X = reduce(hcat, (randn(n_features, samples_per_cluster) .+ Δ .* rand(n_features) for _ in 1:n_clusters))
results = run_cohirf_iterations(X, R=10, C=10, q=9)
results.labels                  # Labels of samples in X
results.n_clusters              # Number of cluster found
results.labels_sequence         # Labels of samples in X at each iteration
results.cluster_representatives # Indices of medoid of cluster
results.n_iterations            # Number of iterations to convergence
```

## Testing the package

In a terminal :
```
cd CoHiRF.jl
julia
```

Inside the Julia REPL, type `]` on the keyboard to access the Pkg interface. Then activate package environment

```
(@v1.12) pkg> activate .
```

Now the command line should look like this 
```
(CoHiRF) pkg>
```

And then simply type
```
(CoHiRF) pkg> test
```

## Launching benchmarks

Benchmarks from the paper were partially reproduced. First with real data from the OpenML datasets :
```
cd CoHiRF.jl
julia --project=benchmark --threads=8 benchmark/openml_benchmarks.jl
```


Then with synthetic data to test for scalability :
```
cd CoHiRF.jl
julia --project=benchmark benchmark/scalability_benchmarks.jl
```

See below for the results.

## Benchmark results

Benchmarks ran on a M4 MacbookPro with 10 CPU 10 GPU 24GB of RAM with no explicit multithreading (for the CoHiRF and KMeans).
Multithreading activated for hyperparameters search.

### Julia benchmarks on Synthetic Data
```
┌   n_samples = 14427
└   n_features = 100
cohirf_ari = = 1.0
CoHiRF runtime
  164.202 ms (89975 allocations: 364.86 MiB)
KMeans runtime
  21.790 ms (159 allocations: 1.66 MiB)

┌   n_samples = 14427
└   n_features = 347
cohirf_ari = = 1.0
CoHiRF runtime
  223.123 ms (90015 allocations: 394.55 MiB)
KMeans runtime
  68.512 ms (159 allocations: 1.73 MiB)

┌   n_samples = 14427
└   n_features = 1202
cohirf_ari = = 1.0
CoHiRF runtime
  416.248 ms (90016 allocations: 507.59 MiB)
KMeans runtime
  235.660 ms (159 allocations: 1.96 MiB)

┌   n_samples = 14427
└   n_features = 4163
cohirf_ari = = 1.0
CoHiRF runtime
  1.127 s (90016 allocations: 819.11 MiB)
KMeans runtime
  1.058 s (159 allocations: 3.13 MiB)

┌   n_samples = 14427
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  3.594 s (90015 allocations: 1.90 GiB)
KMeans runtime
  3.632 s (159 allocations: 5.75 MiB)

┌   n_samples = 14427
└   n_features = 50000
cohirf_ari = = 1.0
CoHiRF runtime
  11.911 s (90016 allocations: 5.72 GiB)
KMeans runtime
  12.795 s (159 allocations: 15.50 MiB)

┌   n_samples = 100
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  2.747 ms (3869 allocations: 12.13 MiB)
KMeans runtime
  24.388 ms (146 allocations: 4.14 MiB)

┌   n_samples = 347
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  11.304 ms (5489 allocations: 40.14 MiB)
KMeans runtime
  86.131 ms (159 allocations: 4.17 MiB)

┌   n_samples = 1202
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  49.314 ms (10633 allocations: 138.11 MiB)
KMeans runtime
  294.977 ms (159 allocations: 4.25 MiB)

┌   n_samples = 4163
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  340.805 ms (28421 allocations: 502.67 MiB)
KMeans runtime
  1.109 s (159 allocations: 4.94 MiB)

┌   n_samples = 14427
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  3.680 s (90016 allocations: 1.97 GiB)
KMeans runtime
  3.881 s (159 allocations: 5.75 MiB)

┌   n_samples = 50000
└   n_features = 14427
cohirf_ari = = 1.0
CoHiRF runtime
  39.504 s (313612 allocations: 9.22 GiB)
KMeans runtime
  14.072 s (159 allocations: 9.41 MiB)
```

### Julia benchmarks on OpenML Datasets

OpenML Datasets (grid search and benchmark using best parameters from paper)
```
 Dataset: ecoli
 Shape: 336 × 7
 Best ARI: 0.8074105509259771 (R=3, C=10, q=25)
  3.362 ms (7946 allocations: 1.14 MiB)

 Dataset: iris
 Shape: 150 × 4
 Best ARI: 0.675280199252802 (R=3, C=4, q=2)
  266.333 μs (2091 allocations: 219.08 KiB)

 Dataset: nursery
 Shape: 12958 × 19
 Best ARI: 0.5425820822941445 (R=3, C=2, q=13)
  51.067 ms (81707 allocations: 49.31 MiB)

 Dataset: satimage
 Shape: 6430 × 36
 Best ARI: 0.6158751586978625 (R=4, C=10, q=21)
  119.405 ms (83290 allocations: 42.13 MiB)

 Dataset: segment
 Shape: 2310 × 16
 Best ARI: 0.5834013699830471 (R=6, C=8, q=3)
  25.745 ms (21026 allocations: 11.88 MiB)

 Dataset: shuttle
 Shape: 58000 × 9
 Best ARI: 0.5860300293354108 (R=6, C=8, q=3)
  811.296 ms (366215 allocations: 1.06 GiB)

 Dataset: alizadeh-2000-v2
 Shape: 62 × 2093
 Best ARI: 0.9460518521129005 (R=2, C=3, q=20)
  372.083 μs (1478 allocations: 1.25 MiB)

 Dataset: garber-2001
 Shape: 66 × 4553
 Best ARI: 0.3832434532345569 (R=2, C=6, q=27)
  1.766 ms (6599 allocations: 6.37 MiB)

 Dataset: har
 Shape: 10299 × 561
 Best ARI: 0.52408345768418 (R=7, C=3, q=9)
  140.379 ms (65990 allocations: 207.95 MiB)

 Dataset: coil-20
 Shape: 1440 × 1024
 Best ARI: 0.5036807138094133 (R=7, C=10, q=5)
  67.661 ms (154476 allocations: 84.90 MiB)
```

### Python benchmarks on OpenML Datasets
For reference, these are the timings on the same machine with the original python code (see https://github.com/BrunoBelucci/cohirf-arxiv/tree/main?tab=readme-ov-file#installing-the-requirements for the installation steps)

Launch the benchmarks using
```
> conda activate cohirf
> cd benchmark
> python openml_benchmarks.py
```

Results :
```
ECOLI
39
R =  8
C =  6
q =  8
CoHiRF ARI =  0.7288141398290872
KMeans ARI =  0.7239733076507628
timings cohirf =  208.32125020679086  ms
timings kmeans =  6.05005023535341  ms

IRIS
61
R =  3
C =  3
q =  26
CoHiRF ARI =  0.6734486404240831
KMeans ARI =  0.6201351808870379
timings cohirf =  26.176916575059295  ms
timings kmeans =  3.0064584920182824  ms

NURSERY
1568
R =  3
C =  4
q =  20
CoHiRF ARI =  0.3305248753473263
KMeans ARI =  0.00020407731830517632
timings cohirf =  173.30073767807335  ms
timings kmeans =  20.091287489049137  ms

SATIMAGE
182
R =  8
C =  8
q =  21
CoHiRF ARI =  0.5313616912366235
KMeans ARI =  0.5661693077461659
timings cohirf =  992.0883998973295  ms
timings kmeans =  44.29193763062358  ms

SEGMENT
40984
R =  10
C =  5
q =  21
CoHiRF ARI =  0.10623573606802778
KMeans ARI =  0.4358155671837634
timings cohirf =  293.7285540625453  ms
timings kmeans =  21.948454179801047  ms

SHUTTLE
40685
R =  8
C =  6
q =  5
CoHiRF ARI =  0.10377959833129029
KMeans ARI =  0.6084038722979527
timings cohirf =  3727.292479155585  ms
timings kmeans =  41.53428738936782  ms

ALIZADEH-2000-V2
46773
R =  3
C =  2
q =  30
CoHiRF ARI =  0.8952131151945433
KMeans ARI =  0.8147059399925529
timings cohirf =  24.81265841051936  ms
timings kmeans =  10.067758266814053  ms

GARBER-2001
46779
R =  10
C =  2
q =  18
CoHiRF ARI =  0.1543835826452616
KMeans ARI =  0.1985708564285454
timings cohirf =  150.52551275584847  ms
timings kmeans =  26.222266629338264  ms

HAR
1478
R =  5
C =  3
q =  23
CoHiRF ARI =  0.33663644768949685
KMeans ARI =  0.4236548520823591
timings cohirf =  471.8862585024908  ms
timings kmeans =  960.9154000179842  ms

COIL-20
46783
R =  7
C =  10
q =  8
CoHiRF ARI =  0.4648567686695173
KMeans ARI =  0.6544739699256954
timings cohirf =  2073.934491397813  ms
timings kmeans =  476.3104375451803  ms
```
