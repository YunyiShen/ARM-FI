# coresets: a package for creating and evaluating logistic regression coresets

The `coresets` package was used produce the experiments for:

[Jonathan H. Huggins](http://www.jhhuggins.org),
[Trevor Campbell](http://www.trevorcampbell.me),
[Tamara Broderick](http://www.tamarabroderick.com).
*[Coresets for Scalable Bayesian Logistic Regression](https://arxiv.org/abs/1605.06423)*.
In *Proc. of the 30th Annual Conference on Neural Information Processing
Systems* (NIPS), 2016.

The package includes functionality to load data, construct coresets
for logistic regression, run an adaptive Metropolis-Hastings sampler using the
coreset, and compare performance of coreset inferences to those obtained with
other methods.

## Compilation and testing

To compile and test the package (for development purposes):
```bash
python setup.py build_ext --inplace  # compile cython code in place
nosetests tests/                     # run tests, which takes a minute or two
```

To install:
```bash
pip install .
```

## Usage

The key function is `coresets.algorithms.construct_lr_coreset_with_kmeans`,
which constructs a logistic regression coreset. The implementation is quite
efficient and requires minimal memory overhead. On a 2012 MacBook Pro with a
2.9 GHz Intel Core i7 it takes about half a second to run on the Webspam
dataset, which contains 350,000 data points of dimension 127.

The script used for most of the experiments in the paper, which makes use of
`construct_lr_coreset_with_kmeans` as well as other functionality, can be found
in
[scripts/run_experiments.py](code/scripts/run_experiments.py).
The script loads data, constructs coresets from the data, and compares the
inferential accuracy of the coreset to a random subsample of the same sizes.
For example, the following would construct random subsets and coresets of
sizes 156, 312, 625, 1250, and 2500 for the Binary10 synthetic dataset.
5,000 iterations of adaptive MALA are used for inference, the radius R is
chosen adaptively, and each experimental condition is repeated 5 times
(running this will take some time -- as much a few hours).
```
lrcoresets$ python scripts/run_experiments.py --synth-bin -d 10 -s 156 312 625 1250 2500 -R=-3 -k=4 -r=5 -i=5000
Created output folder: results
changed working directory to results
Created output folder: ../data/synthetic
Generating data with train path ../data/synthetic/binary-d-10-N-1000000.npz
running experiment...
Created output folder: synthetic_cmc-Ks=4-Ms=156,312,625,1250,2500-Rs=-3.0-d=10-iters=5000-experiment-MALA
X is either low-dimensional or not very sparse, so converting to a numpy array
X is either low-dimensional or not very sparse, so converting to a numpy array
running target algorithm "Full" for 25000 iterations with a warmup of 12500 iterations
File Binary-Full.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=156, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Coreset-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=156.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=312, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Coreset-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=312.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=625, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Coreset-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=625.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=1250, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Coreset-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=1250.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=2500, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Coreset-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=2500.cpk does not yet exist, so starting from scratch
running algorithm "Random" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=156, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Random-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=156.cpk does not yet exist, so starting from scratch
running algorithm "Random" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=312, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Random-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=312.cpk does not yet exist, so starting from scratch
running algorithm "Random" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=625, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Random-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=625.cpk does not yet exist, so starting from scratch
running algorithm "Random" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=1250, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Random-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=1250.cpk does not yet exist, so starting from scratch
running algorithm "Random" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=2500, R=-3.0, K=4, kmeans_subsample_size=auto
File Binary-Random-K=4-R=-3.0-kmeans_subsample_size=auto-output_size_param=2500.cpk does not yet exist, so starting from scratch
plotting experiment results...
Created output folder: synthetic_cmc-Ks=4-Ms=156,312,625,1250,2500-Rs=-3.0-d=10-iters=5000-experiment-MALA_plots
```
The script saves the experimental data and figures the *results/* directory.
Some example plots:

![Polynomial MMD 2 by subset size](data/readme/subset-size-vs-polynomial-mmd-2.png)

*Subset size vs. 2nd degree polynomial MMD for coresets and random subsamples*

![Polynomial MMD 2 by runtime](data/readme/runtime-vs-polynomial-mmd-2.png)

*Running time vs. 2nd degree polynomial MMD for coresets and random subsamples*

![Mean scatter for 312](data/readme/mean-scatter-312.png)

*True vs. estimated means for coresets and random subsamples of size 312*

This invocation of runs a timing experiment on two synthetic datasets and the
three included real datasets:
```
lrcoresets$ python scripts/run_experiments.py --timing --synth-bin --synth-mix --chemreact --webspam --covtype -d 0 -s 156 312 625 1250 2500 5000 10000 -R=-3 -k=6 -r=5 -i=5000
changed working directory to results
Generating data with train path ../data/synthetic/binary-d-10-N-1000000.npz
Generating data with train path ../data/synthetic/mixture-d-10-N-1000000.npy
running experiment...
Created output folder: binary-mixture-chemreact-webspam-covtype-Ks=6-Ms=156,312,625,1250,2500,5000,10000-Rs=-3.0-d=0-iters=5000-timing-experiment-MALA
X is either low-dimensional or not very sparse, so converting to a numpy array
X is either low-dimensional or not very sparse, so converting to a numpy array
running target algorithm "Full" for 2 iterations with a warmup of 1 iterations
File Binary-Full.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=156, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=156.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=312, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=312.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=625, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=625.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=1250, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=1250.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=2500, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=2500.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=5000, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=5000.cpk does not yet exist, so starting from scratch
running algorithm "Coreset" for 5000 iterations with a warmup of 2500 iterations and with the following parameters: output_size_param=10000, R=-3.0, K=6, kmeans_subsample_size=auto
File Binary-Coreset-K=6-R=-3.0-kmeans_subsample_size=auto-output_size_param=10000.cpk does not yet exist, so starting from scratch
.
.
.
plotting experiment results...
Created output folder: binary-mixture-chemreact-webspam-covtype-Ks=6-Ms=156,312,625,1250,2500,5000,10000-Rs=-3.0-d=0-iters=5000-timing-experiment-MALA_plots
```

![Percent time spent creating coreset](data/readme/percent-time-creating-coreset.png)

*Percentage of time spent creating the coreset relative to the total inference
time*
