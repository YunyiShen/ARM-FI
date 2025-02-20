# Authors: Trevor Campbell <tdjc@mit.edu>
#          Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import os
from collections import defaultdict
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import cPickle as cpk
import time

from coresets.data import load_data
from coresets.inference import mh
from coresets.distributions import logistic_likelihood, logistic_likelihood_grad
from coresets.utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class Result(object):
    def disp(self, full=False):
        print('Result:')
        print('Target:', self.target_name)
        print('Algorithm:', self.alg_name)
        print('Dataset:', self.data_name)
        print('# Feats:', self.n_features)
        print('# Data:', self.n_full_data)
        print('Prms:', self.prms)
        if full:
            print('Sampling Times:', self.sample_times)
            print('Algorithm Times:', self.alg_times)
            print('Random States:', self.random_states)
            print('Accept Rates:', self.accept_rates)
            # print('Subsamples:', self.subsamples)
            print('Theta Samples:', self.theta_samples_list)
            print('Weights:', self.weights)
            print('Evaluations:', self.evals)

    def populate(self, target_name, alg_name, data_name,
                 n_features, n_full_data, prms):
        self.target_name = target_name
        self.alg_name = alg_name
        self.data_name = data_name
        self.n_features = n_features
        self.n_full_data = n_full_data
        self.prms = prms
        self.sample_times = []
        self.alg_times = []
        self.random_states = []
        self.accept_rates = []
        # self.subsamples = []
        self.theta_samples_list = []
        self.weights = []
        self.evals = []

    def load(self, fname, check=False):
        try:
            f = open(fname, 'r')
            r = cpk.load(f)
            n_old = len(r.alg_times)
            if (len(r.alg_times) != n_old
                or len(r.sample_times) != n_old
                or len(r.random_states) != n_old
                or len(r.accept_rates) != n_old
                # or len(r.subsamples) != n_old
                or len(r.theta_samples_list) != n_old
                or len(r.weights) != n_old
                or len(r.evals) != n_old):
                print('Error: File with expected name exists, but data is '
                      'corrupt. Removing corrupted data and continuing.')
                print(n_old, len(r.alg_times), len(r.sample_times),
                      len(r.random_states), len(r.accept_rates),
                      len(r.theta_samples_list), len(r.weights), len(r.evals))
            elif check and (r.prms != self.prms
                            or self.target_name != r.target_name
                            or self.alg_name != r.alg_name
                            or self.data_name != r.data_name
                            or self.n_features != r.n_features
                            or self.n_full_data != r.n_full_data):
                print('Error: Requested checking, and the data was '
                      'inconsistent. Ignoring incoming inconsistent data '
                      'and continuing...')
            else:
                self.target_name = r.target_name
                self.alg_name = r.alg_name
                self.data_name = r.data_name
                self.n_features = r.n_features
                self.n_full_data = r.n_full_data
                self.prms = r.prms
                self.sample_times = r.sample_times
                self.alg_times = r.alg_times
                self.random_states = r.random_states
                self.accept_rates = r.accept_rates
                # self.subsamples = r.subsamples
                self.theta_samples_list = r.theta_samples_list
                self.weights = r.weights
                self.evals = r.evals
        except IOError as e:
            print('File %s does not yet exist, so starting from scratch'
                    % os.path.basename(fname))
        else:
            f.close()

    def __eq__(self, r):
        if not isinstance(r, Result):
            return False
        return (self.target_name == r.target_name
                and self.alg_name == r.alg_name
                and self.data_name == r.data_name
                and self.n_features == r.n_features
                and self.n_full_data == r.n_full_data
                and self.prms == r.prms
                and self.sample_times == r.sample_times
                and self.alg_times == r.alg_times
                and self.random_states == r.random_states
                and self.accept_rates == r.accept_rates
                # and self.subsamples == r.subsamples
                and self.theta_samples_list == r.theta_samples_list
                and self.weights == r.weights
                and self.evals == r.evals)


def _is_name_callable_pair_list(value):
    if not isinstance(value, list):
        return False
    pair_checks = [isinstance(v, tuple) and len(v)==2 and callable(v[1])
                   and isinstance(v[0], str) for v in value]
    return all(pair_checks)


def _is_list_of_str_tuples(value, num=3):
    if not isinstance(value, list):
        return False
    tuple_checks = [isinstance(v, tuple) and len(v) == num
                    and all([isinstance(e, str) for e in v]) for v in value]
    return all(tuple_checks)


class Experiment(object):
    def __init__(self, name, algs, prm_dicts, evals, datasets,
                test_datasets=None, target_name='Full', run_target_once=False):
        # check validity of class member inputs
        if test_datasets is None:
            test_datasets = []
        if not _is_name_callable_pair_list(algs):
            raise ValueError("Argument 'algs' in init method of Experiment "
                             "must be a list of 2-tuples of "
                             "('name', callable_function)")
        if (not isinstance(prm_dicts, list)
            or not all( [isinstance(p, dict) for p in prm_dicts])):
           raise ValueError("Argument 'prm_dicts' in init method of Experiment "
                            "must be a dictionary of parameter names mapped "
                            "to values")
        if not _is_name_callable_pair_list(evals):
            raise ValueError("Argument 'evals' in init method of Experiment "
                             "must be a list of 2-tuples of "
                             "('name', callable_function)")
        if not _is_list_of_str_tuples(datasets):
            raise ValueError("Argument 'datasets' in init method of Experiment "
                             "must be a list of 3-tuples of strings "
                             "('name', 'path', 'filetype')")
        if not _is_list_of_str_tuples(test_datasets):
            raise ValueError("Argument 'test_datasets' in init method of "
                             "Experiment must be a list of 3-tuples of strings "
                             "('name', 'path', 'filetype')")
        dnames = [d[0] for d in datasets]
        if not all([t[0] in dnames for t in test_datasets]):
            raise ValueError("In argument 'test_datasets' in init method of "
                             "Experiment, all names of test_datasets must "
                             "correspond to a name in datasets.\nDatasets: "
                             + str(dnames) + "\n Test Datasets: "
                             + str([t[0] for t in test_datasets]))
        if not isinstance(target_name, str):
            raise ValueError("Argument 'target_name' in init method of "
                             "Experiment must be a string")
        if not isinstance(name, str):
            raise ValueError("Argument 'name' in init method of Experiment "
                             "must be a string")
        if target_name not in [a[0] for a in algs]:
            raise ValueError("In init method of Experiment, one of the algs "
                             "must have target_name. Algs: "
                             + str([a[0] for a in algs])
                             + " target_name: " + target_name)
        # populate class members
        self.algs = algs[:]
        self.evals = evals[:]
        self.prm_dicts = prm_dicts[:]
        self.datasets = datasets[:]
        if test_datasets is not None:
            self.test_datasets = test_datasets[:]
        else:
            self.test_datasets = None
        self.target_name = target_name
        self.run_target_once = run_target_once
        self.name = name
        self.target_alg = None
        # find the target algorithm and separate it
        for i in range(len(self.algs)):
            if self.algs[i][0] == target_name:
                self.target_alg = self.algs[i][1]
                self.algs.pop(i)
                break
        assert self.target_alg is not None

    def run(self, n_trials, steps, target_alg_steps=None, thin=1, warmup=None,
            max_dim=0, max_data=0, mh_alg='RW', include_offset=False,
            multithreaded=False, fill_out_features=False):
        ALGS = ['RW', 'MALA']
        if mh_alg not in ALGS:
            raise ValueError("'mh_alg' must be one of " + ','.join(ALGS))
        # make the output folder
        create_folder_if_not_exist(self.name)

        if self.run_target_once:
            n_target_trials = 1
        else:
            n_target_trials = n_trials

        if target_alg_steps is None:
            target_alg_steps = steps
        if warmup is None:
            warmup = max(1, steps / 2)
            target_alg_warmup = max(1, target_alg_steps / 2)
        else:
            target_alg_warmup = max(1, warmup)

        # run the experiment for each dataset
        for dname, dpath, dtype in self.datasets:
            # load training data
            X, y, pp_obj = load_data(dpath, dtype, max_data, max_dim,
                                     True, include_offset)
            D = X.shape[1]
            # load testing data if it exists
            X_test = None
            y_test = None
            for tname, tpath, ttype in self.test_datasets:
                if tname == dname:
                    X_test, y_test, _ = load_data(tpath, ttype, max_data,
                                                  max_dim, pp_obj,
                                                  include_offset)
                    D_test = X_test.shape[1]
                    break
            if X_test is not None:
                if fill_out_features:
                    X, X_test = ensure_dimension_matches(X, X_test)
                elif D != D_test:
                    raise ValueError("X and X_test do not have the same "
                                     "dimensionality: %d vs. %d" %p
                                     (D, D_test))

            # construct the sampler
            if mh_alg == 'RW':
                sampler = self._build_rw_sampler(D, steps, thin, warmup)
                target_sampler = self._build_rw_sampler(D,
                                                        target_alg_steps,
                                                        thin,
                                                        target_alg_warmup)
            elif mh_alg == 'MALA':
                sampler = self._build_mala_sampler(D, steps, thin, warmup)
                target_sampler = self._build_mala_sampler(D,
                                                          target_alg_steps,
                                                          thin,
                                                          target_alg_warmup)
            else:
                raise RuntimeError()  # should never happen
            # run the target algorithm (e.g. the full dataset)
            print('running target algorithm "%s" for' % self.target_name,
                  target_alg_steps, 'iterations with a warmup of',
                  target_alg_warmup, 'iterations')
            target_samples = self._run_alg(dname, self.target_name,
                                           self.target_alg, {}, target_sampler,
                                           n_target_trials,
                                           X, y, X_test, y_test)
            #run comparison algorithms
            for aname, alg in self.algs:
                for prms in self.prm_dicts:
                    print('running algorithm "%s" for' % aname,
                          steps, 'iterations with a warmup of',
                          warmup, 'iterations and with the following',
                          'parameters:',
                          ", ".join(["%s=%s" % item for
                                        item in prms.iteritems()]))
                    self._run_alg(dname, aname, alg, prms, sampler, n_trials,
                                  X, y, X_test, y_test, target_samples,
                                  self.target_name)

    def _build_rw_sampler(self, D, steps, thin, warmup):
        # model is logistic likelihood with Gaussian prior, mean 0 variance 4
        post_p = lambda theta, Z, w: (logistic_likelihood(theta, Z, w)
                                      - .5 * np.sum(theta**2) / 2.**2)
        q = None  # None denotes a symmetric proposal distribution
        samp_q = lambda theta, ell: theta + np.exp(ell) * np.random.randn(D)
        # initial state is 0
        theta0 = np.zeros(D)
        # initial adaptation parameter
        ell0 = np.log(2.38 / np.sqrt(D))
        # return the sampler
        return lambda Z, w: mh(theta0, lambda theta: post_p(theta, Z, w),
                               q, samp_q, steps=steps, warmup=warmup,
                               thin=thin, proposal_param=ell0)

    def _build_mala_sampler(self, D, steps, thin, warmup):
        # model is logistic likelihood with Gaussian prior, mean 0 variance 4
        post_p = lambda theta, Z, w: (logistic_likelihood(theta, Z, w)
                                      - .5 * np.sum(theta**2) / 2.**2)
        def prop_mean(theta, ell, Z, w):
            grad = logistic_likelihood_grad(theta, Z, w)
            return theta + .5 * np.exp(2 * ell) * grad
        def q(theta, theta_new, ell, Z, w):
            diff = prop_mean(theta, ell, Z, w) - theta_new
            return -.5 * np.exp(-2 * ell) * np.sum(diff**2)
        def samp_q(theta, ell, Z, w):
            mean = prop_mean(theta, ell, Z, w)
            return mean + np.exp(ell) * np.random.randn(D)
        # initial state is 0
        theta0 = np.zeros(D)
        # initial adaptation parameter
        ell0 = np.log(2.38 / np.sqrt(D))
        # return the sampler
        return lambda Z, w: mh(
            theta0,
            lambda theta: post_p(theta, Z, w),
            lambda theta, theta_new, ell: q(theta, theta_new, ell, Z, w),
            lambda theta, ell: samp_q(theta, ell, Z, w),
            steps=steps, warmup=warmup, thin=thin,
            proposal_param=ell0, target_rate=.574)


    def _run_alg(self, dname, aname, alg, prms, sampler, n_trials, X, y,
                 X_test, y_test, target_samples=None, target_name=None):
        # preprocess training X and y into z
        if sp.issparse(X):
            Z = sp.diags(y).dot(X)
        else:
            Z = y[:, np.newaxis] * X
        D = X.shape[1]

        # setup the arguments object for calling alg by adding data
        args = prms.copy()
        args['data'] = Z

        # generate the filename
        if len(prms) > 0:
            params_str = pretty_file_string_from_dict(prms)
            namebase = '-'.join([dname, aname, params_str])
        else:
            namebase = '-'.join([dname, aname])
        fname = os.path.join(self.name, namebase  + '.cpk')

        r = Result()
        r.populate(target_name, aname, dname, D, X.shape[0], prms)
        r.load(fname)
        n_old_trials = len(r.alg_times)

        # we don't need any more data
        if n_old_trials >= n_trials:
            if self.run_target_once and self.target_name == aname:
                return r.theta_samples_list
            else:
                return []

        if target_samples is not None:
            if (not self.run_target_once
                and len(target_samples) < n_trials - n_old_trials):
                raise ValueError('The number of samples from the target is not '
                                 'enough for n_trials')

        theta_samples_list = []
        for i in range(n_trials - n_old_trials):
            # save the random state at the beginning of the trial (if replay is
            # needed later)
            random_state = npr.get_state()

            # generate subsample via alg
            t0 = time.clock()
            subsample, weights = call_with_superset_args(alg, args)
            t1 = time.clock()

            # run posterior inference
            theta_samples, accept_rate = sampler(subsample, weights)
            t2 = time.clock()

            # record timing/seed/names/samples/accept rates
            alg_time = t1 - t0
            sample_time = t2 - t1
            theta_samples = np.array(theta_samples)
            theta_samples_list.append(theta_samples)

            # gather results
            r.alg_times.append(alg_time)
            r.sample_times.append(sample_time)
            r.random_states.append(random_state)
            r.accept_rates.append(accept_rate)
            # only record the subsample if it's not the raw data (the target)
            if target_samples is not None:
                # r.subsamples.append(subsample)
                r.weights.append(weights)
            else:
                # r.subsamples.append(None)
                r.weights.append(None)

            # evaluate the posterior sample sets
            evaldict = {}
            for ename, ev in self.evals:
                if target_samples is not None:
                    t_idx = 0 if self.run_target_once else i
                    evaldict[ename] = ev(target_samples[t_idx], theta_samples,
                                         X_test, y_test)
                else:
                    evaldict[ename] = ev(None, theta_samples, X_test, y_test)
            r.evals.append(evaldict)
        r.theta_samples_list.extend(theta_samples_list)
        # save the results
        try:
            f = open(fname, 'w')
            cpk.dump(r, f)
        except IOError as e:
            print('Error: Failure to write results to disk')
            print('Continuing execution anyway...')
        else:
            f.close()
        return theta_samples_list


# Plotting helpers
def _insert_mean_std(w, wselect, r):
    try:
        li = wselect(r)
        w.append([np.mean(li), np.std(li) / np.sqrt(len(li))])
    except:
        w.append([np.nan, np.nan])


def _get_colormap(names, palette_id, custom_map={}):
    unq_names = set(names) - set(custom_map.keys())
    offset = len(custom_map)
    unq_palette = sns.color_palette(palette_id, len(unq_names) + offset)
    cmap = {}
    for i, nm in enumerate(unq_names):
        cmap[nm] = unq_palette[i + offset]
    if custom_map is not None:
        for key, value in custom_map.iteritems():
            cmap[key] = value
    return cmap


def _bar(names, y, y_label, cmap):
    if len(set(names)) != len(names):
        raise ValueError('Bar plot must have only one entry per name')
    width = 1.
    ind0 = 0.
    for i in range(len(names)):
        if y[i, 1] > 0.:
            plt.bar(ind0 + i * width, y[i, 0], width, yerr=y[i, 1],
                    color=cmap[names[i]],
                    ecolor=tuple([v*.3 for v in cmap[names[i]]]))
        else:
            plt.bar(ind0+i*width, y[i, 0], width, color=cmap[names[i]])
    plt.tick_params(axis='x', which='both', bottom='off',
                    top='off', labelbottom='off')
    plt.legend(names, loc=0)
    plt.ylabel(y_label)


def _scatter(names, x, y, x_label, y_label, cmap, xscale, yscale, legend_kwargs, line=False):
    hndls = []
    unique_names_list = list(set(names))
    for nm in unique_names_list:
        idces = [i for i, n in enumerate(names) if n == nm]
        xnm = x[idces, :]
        ynm = y[idces, :]
        srtidcs = xnm[:, 0].argsort()
        xnm = xnm[srtidcs, :]
        ynm = ynm[srtidcs, :]
        if np.any(np.isnan(xnm)):
            if len(idces) > 1:
                raise ValueError('hline cannot be drawn with multiple y values '
                                 'for same name')
            h = plt.axhline(ynm[0, 0], c=cmap[nm])
            plt.axhline(ynm[0, 0]+ynm[0, 1], c=cmap[nm], linestyle='--')
            plt.axhline(ynm[0, 0]-ynm[0, 1], c=cmap[nm], linestyle='--')
        elif np.any(np.isnan(ynm)):
            if len(idces) > 1:
                raise ValueError('vline cannot be drawn with multiple y values '
                                 'for same name')
            h = plt.axvline(xnm[0, 0], c=cmap[nm])
            plt.axvline(xnm[0, 0]+xnm[0, 1], c=cmap[nm], linestyle='--')
            plt.axvline(xnm[0, 0]-xnm[0, 1], c=cmap[nm], linestyle='--')
        else:
            if line:
                h, = plt.plot(xnm[:, 0], ynm[:, 0], c=cmap[nm])
            else:
                h = plt.scatter(xnm[:, 0], ynm[:, 0], c=cmap[nm])
            if np.any(x[idces, 1] > 0.):
                plt.errorbar(xnm[:, 0], ynm[:, 0], xerr=xnm[:, 1], c=cmap[nm],
                             linestyle="None")
            if np.any(y[idces, 1] > 0.):
                plt.errorbar(xnm[:, 0], ynm[:, 0], yerr=ynm[:, 1], c=cmap[nm],
                             linestyle="None")
        hndls.append(h)
    plt.legend(hndls, unique_names_list, **legend_kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    plt.yscale(yscale)


def _line(names, x, y, x_label, y_label, cmap, xscale, yscale, legend_kwargs):
    _scatter(names, x, y, x_label, y_label, cmap, xscale, yscale, legend_kwargs,
             line=True)


def _contour(names, x, y, z, x_label, y_label, z_label, cmap):
    raise NotImplementedError('Contour not implemented yet!')


def plot_results(dirname, x_selector=None, y_selector=None, z_selector=None,
                 x_label=None, y_label=None, z_label=None, plot_type='bar',
                 xscale='linear', yscale='linear', legend_kwargs={'loc' : 0},
                 alg_to_color=None, data_to_color=None, show=False,
                 excluded_algs=[]):
    if x_selector is not None and x_label is None:
        raise ValueError('Need an axis label for x selector')
    if y_selector is not None and y_label is None:
        raise ValueError('Need an axis label for y selector')
    if z_selector is not None and z_label is None:
        raise ValueError('Need an axis label for z selector')

    out_dirname = dirname + '_plots'
    create_folder_if_not_exist(out_dirname)
    labels_str = '-'.join([label for label in [x_label, y_label, z_label]
                                if label is not None])

    # for each dataset/algorithm, load the x, y, and z
    # if there is a list of entries of length > 1, do errorbars
    alg_names = []
    data_names = []
    x = []
    y = []
    z = []

    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})

    # iterator over results files
    for fn in os.listdir(dirname):
        if not fn.endswith('.cpk'):
            continue
        with open(os.path.join(dirname, fn), 'r') as f:
            r = cpk.load(f)
        if r.alg_name in excluded_algs:
            continue
        alg_names.append(r.alg_name)
        data_names.append(r.data_name)
        # get x/y/z
        if x_selector is not None:
            _insert_mean_std(x, x_selector, r)
        if y_selector is not None:
            _insert_mean_std(y, y_selector, r)
        if z_selector is not None:
            _insert_mean_std(z, z_selector, r)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # assign colors to alg names and dataset names
    alg_cmap = _get_colormap(alg_names, 'hls', alg_to_color)
    data_cmap = _get_colormap(data_names, 'hls', data_to_color)

    # if a single alg/dataset, just plot (no legend; default to algorithm color)
    for an in set(alg_names):
        anidces = [i for i, a in enumerate(alg_names) if a == an]
        da = [data_names[i] for i in anidces]
        xa = x[anidces, :] if x.size > 0 else None
        ya = y[anidces, :] if y.size > 0 else None
        za = z[anidces, :] if z.size > 0 else None
        plt.figure()
        plt.clf()
        if plot_type == 'bar':
            _bar(da, ya, y_label, data_cmap)
        elif plot_type == 'line':
            _line(da, xa, ya, x_label, y_label, data_cmap, xscale, yscale,
                  legend_kwargs)
        elif plot_type == 'scatter':
            _scatter(da, xa, ya, x_label, y_label, data_cmap, xscale, yscale,
                     legend_kwargs)
        elif plot_type == 'contour':
            _contour(da, xa, ya, za, x_label, y_label, z_label, data_cmap)
        else:
            raise ValueError('Plot type not recognized')
        sns.despine()
        name_parts = [an, labels_str, plot_type, xscale, yscale]
        plot_name =  '-'.join(name_parts) + '.pdf'
        plt.savefig(os.path.join(out_dirname, plot_name),
                    bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    if not (len(set(data_names)) == 1 and len(set(alg_names)) == 1):
        for dn in set(data_names):
            dnidces = [i for i, d in enumerate(data_names) if d == dn]
            ad = [alg_names[i] for i in dnidces]
            xd = x[dnidces, :] if x.size > 0 else None
            yd = y[dnidces, :] if y.size > 0 else None
            zd = z[dnidces, :] if z.size > 0 else None
            plt.figure()
            plt.clf()
            if plot_type == 'bar':
                _bar(ad, yd, y_label, alg_cmap)
            elif plot_type == 'line':
                _line(ad, xd, yd, x_label, y_label, alg_cmap, xscale, yscale,
                      legend_kwargs)
            elif plot_type == 'scatter':
                _scatter(ad, xd, yd, x_label, y_label, alg_cmap, xscale, yscale,
                         legend_kwargs)
            elif plot_type == 'contour':
                _contour(ad, xd, yd, zd, x_label, y_label, z_label, alg_cmap)
            else:
                raise ValueError('Plot type not recognized')
            sns.despine()
            name_parts = [dn, labels_str, plot_type, xscale, yscale]
            plot_name =  '-'.join(name_parts) + '.pdf'
            plt.savefig(os.path.join(out_dirname, plot_name),
                        bbox_inches='tight')
            if show:
                plt.show()
            plt.close()
    return alg_cmap, data_cmap


def plot_means_and_vars(dirname, target_name, selector_name, selector,
                        alg_to_color=None, show=False):
    out_dirname = dirname + '_plots'
    create_folder_if_not_exist(out_dirname)

    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})

    samples = {}
    data_names = set()
    alg_names = set()
    selected_values = set()

    for fn in os.listdir(dirname):
        if not fn.endswith('.cpk'):
            continue
        with open(os.path.join(dirname, fn), 'r') as f:
            r = cpk.load(f)
        if r.alg_name != target_name:
            key = (r.alg_name, r.data_name, selector(r))
            selected_values.add(selector(r))
            alg_names.add(r.alg_name)
            data_names.add(r.data_name)
        else:
            key = (r.alg_name, r.data_name)
        samples[key] = r.theta_samples_list

    alg_cmap = _get_colormap(alg_names, 'hls', alg_to_color)
    num_algs = len(alg_names)

    ylabel = target_name
    plot_type_info = [(np.mean, 'mean', 'linear'),
                      (np.var, 'variance', 'log')]
    for data_name in data_names:
        target_theta_samples = samples[(target_name, data_name)][0]
        target_funs = {}
        target_funs['mean'] = np.mean(target_theta_samples, 0)
        target_funs['variance'] = np.var(target_theta_samples, 0)
        for value in list(selected_values):
            for fun, fun_name, xyscale in plot_type_info:
                plt.figure()
                plt.clf()
                for i, alg in enumerate(alg_names):
                    theta_samples = np.array(samples[(alg, data_name, value)])
                    all_alg_fun = fun(theta_samples, 1)
                    scale = np.sqrt(all_alg_fun.shape[1])
                    alg_fun_mean = np.mean(all_alg_fun, 0)
                    alg_fun_err = np.std(all_alg_fun, 0) / scale
                    if np.any(alg_fun_err > 0):
                        plt.errorbar(target_funs[fun_name], alg_fun_mean,
                                     yerr=alg_fun_err,
                                     c='k', linewidth=1, linestyle="None")
                    plt.scatter(target_funs[fun_name], alg_fun_mean,
                                c=alg_cmap[alg])
                plt.xscale(xyscale)
                plt.yscale(xyscale)
                plt.legend(alg_names, loc=0)
                xmin, xmax, ymin, ymax = plt.axis()
                new_min = min(xmin, ymin)
                new_max = max(xmax, ymax)
                start = 1.01 * max(xmin, ymin)
                end = 0.99 * min(xmax, ymax)
                plt.plot([start, end], [start, end], '-k', alpha=.5)
                plt.xlabel('True %ss' % fun_name)
                plt.ylabel('Approximate %ss' % fun_name)
                sns.despine()
                plt.axis((xmin, xmax, ymin, ymax))
                #plt.axis((new_min, new_max, new_min, new_max))
                name_parts = [fun_name, 'scatter', data_name,
                              selector_name, str(value)]
                plot_name =  '-'.join(name_parts) + '.pdf'
                plt.savefig(os.path.join(out_dirname, plot_name),
                            bbox_inches='tight')
                if show:
                    plt.show()
                plt.close()
    return alg_cmap


def plot_subsample(dname, dtype, fname, x_comp, y_comp, subsample_index=0,
                   max_data=0, max_dim=0, show=False):
    # load the full dataset
    X, y = load_data(dname, dtype, max_data, max_dim, True)
    D = X.shape[1]
    Z = y[:, np.newaxis]*X
    # load the subsampled data
    try:
        f = open(fname, 'r')
        r = cpk.load(f)
    except IOError as e:
        print('Error: Failure to read results from disk')
        raise
    else:
        f.close()
    Zsub = r.subsamples[subsample_index]
    wtsub = r.weights[subsample_index]
    ysub = np.zeros(Zsub.shape[0])

    # pull the y values from the closest datapoint in the dataset
    for i in range(Zsub.shape[0]):
        distsqs = ((Z - Zsub[i, :])**2).sum(axis=1)
        ysub[i] = y[distsqs.argmin()]
    Xsub = ysub[:, np.newaxis]*Zsub

    # scale the weights
    wtsub = 20 * 4**((wtsub - wtsub.min())/(wtsub.max()-wtsub.min()))

    # now plot
    rd = sns.xkcd_rgb['pale red']
    bl = sns.xkcd_rgb['medium blue']
    plt.figure()
    plt.clf()
    plt.scatter(X[y == 1, x_comp], X[y == 1, y_comp], c=bl, alpha=0.1)
    plt.scatter(X[y == -1, x_comp], X[y == -1, y_comp], c=rd, alpha=0.1)
    plt.scatter(Xsub[ysub == 1, x_comp], Xsub[ysub == 1, y_comp], c=bl,
                s=wtsub[ysub==1])
    plt.scatter(Xsub[ysub == -1, x_comp], Xsub[ysub == -1, y_comp], c=rd,
                s=wtsub[ysub==-1])
    plt.xlabel('Component '+str(x_comp))
    plt.ylabel('Component '+str(y_comp))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(r.alg_name+'-'+r.data_name+'.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
