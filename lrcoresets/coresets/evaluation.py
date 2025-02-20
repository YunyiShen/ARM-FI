# Authors: Jonathan Huggins <jhuggins@mit.edu>
#          Trevor Campbell <tdjc@mit.edu>

from __future__ import absolute_import, print_function

import numpy as np
import scipy.sparse as sp

from coresets.kernels import PolynomialKernel
from coresets.distributions import logistic_likelihood


def _mean_errors(target_samples, other_samples):
    return np.abs(np.mean(target_samples, 0) - np.mean(other_samples, 0))

def _variance_errors(target_samples, other_samples):
    return np.abs(np.var(target_samples, 0) - np.var(other_samples, 0))

def median_mean_error(target_samples, other_samples, X_test, y_test):
    if target_samples is None:
        return 0.0
    return np.median(_mean_errors(target_samples, other_samples))

def mean_mean_error(target_samples, other_samples, X_test, y_test):
    if target_samples is None:
        return 0.0
    return np.mean(_mean_errors(target_samples, other_samples))

def median_variance_error(target_samples, other_samples, X_test, y_test):
    if target_samples is None:
        return 0.0
    return np.median(_variance_errors(target_samples, other_samples))

def mean_variance_error(target_samples, other_samples, X_test, y_test):
    if target_samples is None:
        return 0.0
    return np.mean(_variance_errors(target_samples, other_samples))


def mean_relative_squared_error(target_samples, other_samples, X_test, y_test):
    """Error in the mean of the samples, normalized by the variance

    For each feature dimension, calculates the squared error in means of then
    samples, normalized by the variance of `target_samples` along that
    dimension. The mean of these relative squared errors is returned.

    Parameters
    ----------
    target_samples : array-like matrix, shape=(n_t_samples, n_features)
        Maye be None, indicating that other_samples is its own target

    other_samples : array-like matrix, shape=(n_o_samples, n_features)

    X_test : unused

    y_test : unused

    Returns
    -------
    mrse : float
    """

    if target_samples is None:
        return 0.0

    assert len(target_samples.shape) == 2
    assert len(other_samples.shape) == 2
    assert target_samples.shape[1] == other_samples.shape[1]

    t_means = np.mean(target_samples, 0)
    o_means = np.mean(other_samples, 0)
    o_vars = np.var(target_samples, 0)
    return np.mean((t_means - o_means)**2 / o_vars)


def log_likelihood(target_samples, other_samples, X_test, y_test):
    """Log likelihood of a test dataset under the posterior theta samples

    Returns the mean log likelihood of the test data over all test datapoints
    and theta samples.

    Parameters
    ----------
    target_samples : unused

    other_samples : array-like matrix, shape=(n_theta_samples, n_features)

    X_test :  array-like matrix, shape=(n_test_data, n_features)

    y_test : array-like matrix, shape=(n_test_data,)

    Returns
    -------
    mean_log_likelihood : float
    """
    assert len(other_samples.shape) == 2
    assert len(X_test.shape) == 2
    assert len(y_test.shape) == 1
    assert X_test.shape[1] == other_samples.shape[1]
    assert y_test.shape[0] == X_test.shape[0]

    if sp.issparse(X_test):
        Z_test = sp.diags(y_test).dot(X_test)
    else:
        Z_test = y_test[:, np.newaxis] * X_test

    ll = logistic_likelihood(other_samples.T, Z_test)
    return ll / (other_samples.shape[0] * X_test.shape[0])


def prediction_error(target_samples, other_samples, X_test, y_test):
    """Prediction error on a test dataset under the posterior theta sampling
    distribution.

    Computes the distribution on the error rate (in [0, 1]) on the test set
    using the label prediction rule based on maximum log likelihood
    under the posterior theta sampling distribution.

    The mean of this distribution is the probability of making an error when
    sampling a single datapoint uniformly from the test set, sampling a value
    of theta from the posterior set of samples, and using the max log
    likelihood prediction rule.

    Parameters
    ----------
    target_samples : unused

    other_samples : array-like matrix, shape=(n_theta_samples, n_features)

    X_test :  array-like matrix, shape=(n_test_data, n_features)

    y_test : array-like matrix, shape=(n_test_data,)

    Returns
    -------
    error_distribution : array-like, shape=(n_theta_samples,)
    """
    assert len(other_samples.shape) == 2
    assert len(X_test.shape) == 2
    assert len(y_test.shape) == 1
    assert X_test.shape[1] == other_samples.shape[1]
    assert y_test.shape[0] == y_test.shape[0]

    # compute the likelihood of each datapoint under assumed label in {1, -1}
    loglikep = logistic_likelihood(other_samples.T, X_test, sum_result=False)
    logliken = logistic_likelihood(other_samples.T, -X_test, sum_result=False)
    # make predictions based on max log likelihood under each sampled parameter
    # theta
    predictions = np.ones(loglikep.shape)
    predictions[logliken > loglikep] = -1
    #compute the distribution of the error rate using max LL on the test set
    # under the posterior theta distribution
    error_val = np.mean(y_test[:, np.newaxis] != predictions)
    return error_val


def make_weighted_brier_score(positive_weight=1.0):
    return (lambda target_samp, other_samp, X_test, y_test:
            weighted_brier_score(other_samp, X_test, y_test, positive_weight))


def weighted_brier_score(samples, X_test, y_test, positive_weight=1.0):
    assert len(samples.shape) == 2
    assert len(X_test.shape) == 2
    assert len(y_test.shape) == 1
    assert X_test.shape[1] == samples.shape[1]
    assert y_test.shape[0] == y_test.shape[0]

    if sp.issparse(X_test):
        Z_test = sp.diags(y_test).dot(X_test)
    else:
        Z_test = y_test[:, np.newaxis] * X_test

    ll = logistic_likelihood(samples.T, Z_test, sum_result=False)
    test_probs = np.mean(np.exp(ll), axis=1)
    test_probs[y_test == 1] -= 1
    scores = test_probs**2
    if positive_weight != 1.0:
        scores[y_test == 1] *= positive_weight
    return np.mean(scores)


def make_polynomial_mmd_evaluation(degree=3, theta=None):
    """Make a Polynomial MMD evaluation function

    See coresets.kernels.PolynomialKernel for a description the polynomial
    kernel parameters.

    Parameters
    ----------
    degree : int, optional

    theta : float, optional

    Returns
    -------
    poly_mmd_eval : function with same signiture as `prediction_error`
    """
    def poly_mmd_eval(target_samples, other_samples, X_test, y_test):
        if target_samples is None:
            return 0.0
        return polynomial_mmd(target_samples, other_samples, degree, theta)
    return poly_mmd_eval


def polynomial_mmd(sample1, sample2, degree, theta=None, test=False,
                   unbiased=False):
    """Calculate Polynomial MMD between samples, and optionally compute p-value

    See coresets.kernels.PolynomialKernel for a description the polynomial
    kernel parameters.

    Parameters
    ----------
    sample1 : matrix-like array, shape=(n_samples1, n_features)

    sample2 : matrix-like array, shape=(n_samples2, n_features)

    degree : int

    theta : float, optional

    test : boolean, optional
        Compute p-value. Default is False.

    unbiased : boolean, optional
        Compute an unbiased estimate of the MMD. Default is False.

    Returns
    -------
    mmd : float

    p_value : float
        Only returned if `test = True`
    """
    if theta is None:
        kernel = PolynomialKernel(degree)
    else:
        kernel = PolynomialKernel(degree, theta)

    # TODO subsample if running a two sample test and the samples are too big;
    # otherwise the two sample test will be way too slow!

    mmd = kernel.estimate_mmd(sample1, sample2, unbiased)

    if test:
        p_value = kernel.two_sample_test(sample1, sample2, 1000)
        return mmd, p_value
    else:
        return mmd
