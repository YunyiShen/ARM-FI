# Authors: Jonathan Huggins <jhuggins@mit.edu>
#          Trevor Campbell <tdjc@mit.edu>

from __future__ import absolute_import, print_function

import sys
import csv
from warnings import warn

import numpy as np
import numpy.random as npr
import scipy.sparse as sp

import sklearn.datasets as skl_ds
from sklearn import preprocessing
from coresets.distributions import logistic_likelihood

import h5py

# based on: http://stackoverflow.com/questions/8955448/
def save_sparse_Xy(filename, X, y):
    """Save sparse X and array-like y as an npz file.

    Parameters
    ----------
    filename : string

    X : sparse matrix, shape=(n_samples, n_features)

    y : array-like, shape=(n_samples,)
    """
    np.savez(filename, data=X.data, indices=X.indices, indptr=X.indptr,
             shape=X.shape, y=y)


def save_Xy(filename, X, y):
    """Save X, y as an npz file.

    Parameters
    ----------
    filename : string

    X : matrix-like, shape=(n_samples, n_features)

    y : array-like, shape=(n_samples,)
    """
    if sp.issparse(X):
        save_sparse_Xy(filename, X, y)
    else:
        np.savez(filename, X=X, y=y)


def _load_svmlight_data(path):
    X, y = skl_ds.load_svmlight_file(path)
    return X, y


def _load_npy_data(path):
    xy = np.load(path)
    X = xy[:, :-1]
    y = xy[:, -1]
    return X, y


def _load_npz_data(path):
    loader = np.load(path)
    if 'X' in loader:
        X = loader['X']
    else:
        X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])
    y = loader['y']
    return X, y


def _load_hdf5_data(path):
    f = h5py.File(path, 'r')
    X = f['x']
    y = f['y']
    f.close()
    return X, y


def _load_csv_data(path):
    xy = np.genfromtxt(path, delimiter=',')
    X = xy[:, :-1]
    y = xy[:, -1]
    return X, y


def load_data(path, file_type, max_data=0, max_dim=0,
              preprocess=True, include_offset=False):
    """Load data from a variety of file types.

    Parameters
    ----------
    path : string
        Data file path.

    file_type : string
        Supported file types are: 'svmlight', 'npy' (with the labels y in the
        rightmost col), 'npz', 'hdf5' (with datasets 'x' and 'y'), and 'csv'
        (with the labels y in the rightmost col)

    max_data : int
        If positive, maximum number of data points to use. If zero or negative,
        all data is used. Default is 0.

    max_dim : int
        If positive, maximum number of features to use. If zero or negative,
        all features are used. Default is 0.

    preprocess : boolean or Transformer, optional
        Flag indicating whether the data should be preprocessed. For sparse
        data, the features are scaled to [-1, 1]. For dense data, the features
        are scaled to have mean zero and variance one. Default is True.

    include_offset : boolean, optional
        Flag indicating that an offset feature should be added. Default is
        False.

    Returns
    -------
    X : array-like matrix, shape=(n_samples, n_features)

    y : int ndarray, shape=(n_samples,)
        Each entry indicates whether each example is negative (-1 value) or
        positive (+1 value)

    pp_obj : None or Transformer
        Transformer object used on data, or None if ``preprocess=False``
    """
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if file_type in ["svmlight", "svm"]:
        X, y = _load_svmlight_data(path)
    elif file_type == "npy":
        X, y = _load_npy_data(path)
    elif file_type == "npz":
        X, y = _load_npz_data(path)
    elif file_type == "hdf5":
        X, y = _load_hdf5_data(path)
    elif file_type == "csv":
        X, y = _load_csv_data(path)
    else:
        raise ValueError("unsupported file type, %s" % file_type)

    y_vals = set(y)
    if len(y_vals) != 2:
        raise ValueError('Only expected y to take on two values, but instead'
                         'takes on the values ' + ', '.join(y_vals))
    if 1.0 not in y_vals:
        raise ValueError('y does not take on 1.0 as one on of its values, but '
                         'instead takes on the values ' + ', '.join(y_vals))
    if -1.0 not in y_vals:
        y_vals.remove(1.0)
        print('converting y values of %s to -1.0' % y_vals.pop())
        y[y != 1.0] = -1.0

    if preprocess is False:
        pp_obj = None
    else:
        if preprocess is True:
            if sp.issparse(X):
                pp_obj = preprocessing.MaxAbsScaler(copy=False)
            else:
                pp_obj = preprocessing.StandardScaler(copy=False)
            pp_obj.fit(X)
        else:
            pp_obj = preprocess
        X = pp_obj.transform(X)

    if include_offset:
        X = preprocessing.add_dummy_feature(X)

    if sp.issparse(X) and (X.nnz > np.prod(X.shape) / 10 or X.shape[1] <= 20):
        print("X is either low-dimensional or not very sparse, so converting "
              "to a numpy array")
        X = X.toarray()
    if isinstance(max_data, int) and max_data > 0 and max_data < X.shape[0]:
        X = X[:max_data,:]
        y = y[:max_data]
    if isinstance(max_dim, int) and max_dim > 0 and max_dim < X.shape[1]:
        X = X[:,:max_dim]

    return X, y, pp_obj


def _generate_and_save_from_X(X, theta, fname):
    lp = logistic_likelihood(theta, X, sum_result=False)
    ln = logistic_likelihood(theta, -X, sum_result=False)
    lmax = np.maximum(lp, ln)
    lp -= lmax
    ln -= lmax
    p = np.exp(lp) / (np.exp(lp) + np.exp(ln))
    y = npr.rand(X.shape[0])
    y[y <= p] = 1
    y[y != 1] = -1
    if fname is not None:
        if sp.issparse(X):
            save_sparse_Xy(fname, X, y)
        else:
            np.save(fname, np.hstack((X, y[:, np.newaxis])))
    return X, y


def _ensure_means_covar_match(means, covar):
    if len(means.shape) == 1:
        n_features = means.shape[0]
    else:
        n_features = means.shape[1]
    if len(covar.shape) != 2 or covar.shape[0] != covar.shape[1]:
        raise ValueError('invalid covariance matrix shape')
    if n_features != covar.shape[0]:
        raise ValueError('mean and covariance shapes do not match')


def generate_gaussian_synthetic(num_samples, mean, covar, theta,
                                fname=None, include_offset=False):
    """Generate classification data with covariates from Gaussian distribution.

    Generate `num_samples` data points with `X[i,:] ~ N(mean, covar)`, then use
    a logistic likelihood model with parameter `theta` to generate `y[i]`.
    If `include_offset = True`, then `X[i,-1] = 1`. Thus,
    `total_features = n_features` if `include_offset = False` and
    `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    mean : array-like, shape=(n_features,)

    covar : matrix-like, shape=(n_features, n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : ndarray with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(mean, covar)
    X = npr.multivariate_normal(mean, covar, num_samples)
    if include_offset:
        X = np.hstack((X, np.ones((num_samples, 1))))
    return _generate_and_save_from_X(X, theta, fname)


def generate_gaussian_mixture(num_samples, weights, means, covar, theta,
                              fname=None, include_offset=False):
    """Generate classification data with covariates from Gaussian mixture.

    Generate `num_samples` data points with `X[i,:] ~ N(means[j,:], covar)`
    with probability `weights[j]`, then use a logistic likelihood model with
    parameter `theta` to generate `y[i]`.  If `include_offset = True`,
    then `X[i,-1] = 1`.  Thus, `total_features = n_features` if
    `include_offset = False` and `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    weights : array-like, shape=(n_components,)

    means : array-like, shape=(n_components, n_features)

    covar : matrix-like, shape=(n_features, n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : ndarray with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(means, covar)
    if means.shape[0] != weights.shape[0]:
        raise ValueError("'means' and 'weights' shapes do not match")
    components = npr.choice(weights.shape[0], num_samples, p=weights)
    z = np.zeros(means.shape[1])
    X = means[components, :] + npr.multivariate_normal(z, covar, num_samples)
    if include_offset:
        X = np.hstack((X, np.ones((num_samples, 1))))
    return _generate_and_save_from_X(X, theta, fname)


def generate_reverse_mixture(num_samples, pos_prob, means, covar, fname=None):
    """Generate classification data class first, then Gaussian covariates.

    Generate `num_samples` data points with `Pr[y[i] = 1] = pos_prob` and
    `X[i,:] ~ N(means[y[i],:], covar)`.

    Parameters
    ----------
    num_samples : int

    pos_prob : float

    means : array-like, shape=(2, n_features)

    covar : matrix-like, shape=(n_features, n_features)

    fname : string, optional
        If provided, save data to the provided filename

    Returns
    -------
    X : ndarray with shape (num_samples, n_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(means, covar)
    if means.shape[0] != 2:
        raise ValueError("'means' must have exactly two means")
    y = npr.rand(num_samples)
    y[y <= pos_prob] = 1
    y[y != 1] = -1
    components = np.zeros(num_samples, dtype=np.int)
    components[y == 1] = 1
    z = np.zeros(means.shape[1])
    X = means[components, :] + npr.multivariate_normal(z, covar, num_samples)
    if fname is not None:
        np.save(fname, np.hstack((X, y[:, np.newaxis])))
    return X, y


def generate_binary_data(num_samples, probs, theta,
                         fname=None, include_offset=False):
    """Generate classification data with binary covariates.

    Generate `num_samples` data points with `Pr[X[i,j] = 1] = probs[j]` and
    a logistic likelihood model with parameter `theta` to generate `y[i]`.
    If `include_offset = True`,  then `X[i,-1] = 1`.  Thus,
    `total_features = n_features` if `include_offset = False` and
    `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    probs : array-like, shape=(n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : csr_matrix with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    probs = probs[np.newaxis, :]
    X = npr.rand(num_samples, probs.shape[1])
    X[X <= probs] = 1
    X[X != 1] = 0
    X = sp.csr_matrix(X, dtype=np.int32)
    if include_offset:
        X = sp.hstack((X, np.ones((num_samples, 1), dtype=np.int32)),
                      format='csr')
    return _generate_and_save_from_X(X, theta, fname)
