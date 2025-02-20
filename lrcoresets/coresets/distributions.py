# Author: Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import numpy as np
import numpy.random as npr
import scipy.sparse as sp


def log_spherical_gaussian(theta, variance):
    return -np.sum(theta**2) / (2 * variance)


def log_spherical_gaussian_grad(theta, variance):
    return -theta / variance


def logistic_likelihood(theta, Z, weights=None, sum_result=True):
    """(Weighted) logistic regression likelihood function.

    Parameters
    ----------
    theta : array-like, shape=(n_features,)

    Z : array-like matrix, shape=(n_samples, n_features)

    weights : array-like, shape=(n_samples,), optional
        Default is None.

    sum_result : boolean, optional
        Default is True.

    Returns
    -------
    likelihood : float or ndarray of floats with shape=(n_samples,)
    """
    if not sp.issparse(Z):
        Z = np.atleast_2d(Z)
    with np.errstate(over='ignore'):  # suppress exp overflow warning
        likelihoods = -np.log1p(np.exp(-Z.dot(theta)))
    if not sum_result:
        return likelihoods
    if weights is not None:
        likelihoods = weights * likelihoods.T
    return np.sum(likelihoods)


def logistic_likelihood_grad(theta, Z, weights=None):
    """Gradient of (weighted) logistic regression likelihood function.

    Parameters
    ----------
    theta : array-like, shape=(n_features,)

    Z : array-like matrix, shape=(n_samples, n_features)

    weights : array-like, shape=(n_samples,), optional
        Default is None.

    Returns
    -------
    gradient : ndarray of floats with shape=(n_features,)
    """
    if not sp.issparse(Z):
        Z = np.atleast_2d(Z)
    grad_weights = 1. / (1. + np.exp(Z.dot(theta)))
    if weights is not None:
        grad_weights *= weights
    if sp.issparse(Z):
        return sp.csr_matrix(grad_weights).dot(Z).toarray().squeeze()
    else:
        return grad_weights.dot(Z)


def logistic_likelihood_hessian(theta, Z, weights=None):
    """Hessian of (weighted) logistic regression likelihood function.

    Parameters
    ----------
    theta : array-like, shape=(n_features,)

    Z : array-like matrix, shape=(n_samples, n_features)

    weights : array-like, shape=(n_samples,), optional
        Default is None.

    Returns
    -------
    hessian : ndarray of floats with shape=(n_features, n_features)
    """
    num_samples, num_features = Z.shape
    if not sp.issparse(Z):
        Z = np.atleast_2d(Z)
    expZtheta = np.exp(Z.dot(theta))
    hessian_weights = expZtheta / (1. + expZtheta)**2
    if weights is not None:
        hessian_weights *= weights
    hessian = np.zeros((num_features, num_features))
    for i in xrange(num_samples):
        hessian += hessian_weights[i] * np.outer(Z[i,:], Z[i,:])
    return hessian
