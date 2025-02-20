# Author: Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import numpy as np
import numpy.random as npr
import scipy.linalg as la
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import k_means

# relying on these is a bit dangerous, but why not live life on the edge?
from sklearn.utils.extmath import row_norms
from sklearn.utils import as_float_array

from . import _algorithms

MAX_CORESET_PROPORTION = 0.5


def _calculate_cluster_information(data, cluster_centers,
                                   cluster_assignments=None):
    """Calculate cluster information for data.

    Only calculates cluster sizes if `cluster_assignments` is provided.

    Parameters
    ----------
    data : array-like matrix, shape=(n_samples, n_features)

    cluster_centers: array-like matrix, shape=(n_clusters, n_features)

    cluster_assignments: int array-like, shape=(n_samples,), optional

    Returns
    -------
    cluster_sizes : int ndarray with shape (n_clusters,)
        number of data points assigned (i.e. closest) to each cluster

    cluster_means : float ndarray with shape (n_clusters, n_features)
        averages of the data assigned to each cluster

    cluster_assignments: float ndarray with shape (coreset_size,)
        cluster assignments for each data point
    """
    cluster_sizes = np.zeros(cluster_centers.shape[0], dtype=np.int32)
    if cluster_assignments is None:
        cluster_assignments = -np.ones(data.shape[0], dtype=np.int32)
        dists = euclidean_distances(data, cluster_centers, squared=True)
        np.argmin(dists, axis=1, out=cluster_assignments)

    if cluster_assignments.dtype != np.int32:
        cluster_assignments = np.array(cluster_assignments, dtype=np.int32)
    unique, unique_counts = np.unique(cluster_assignments,
                                      return_counts=True)
    assert unique[-1] <= cluster_sizes.shape[0] - 1
    cluster_sizes[unique] = unique_counts

    return cluster_sizes, cluster_centers, cluster_assignments


def _calculate_assignments(data, centers):
     cluster_assignments = -np.ones(data.shape[0], dtype=np.int32)
     dists = euclidean_distances(data, centers, squared=True)
     np.argmin(dists, axis=1, out=cluster_assignments)
     inertia = np.sum(np.amin(dists, axis=1))
     _, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
     return cluster_sizes, cluster_assignments, inertia


def _calculate_lr_sensitivities_fast(data, R, ball_center, cluster_info):
    """Approximately calculate the sensitivities of each data point for logistic
    regression.

    Not exact because the centers are not adjusted for each data point.
    In practice the relative error should only be a few perfect and
    runtime is improved by ~2-8x compared to the cython version and ~100-150x
    compared to the exact python version.
    """
    cluster_sizes, cluster_means, cluster_assignments = cluster_info
    num_samples, num_features = data.shape
    num_clusters = cluster_sizes.shape[0]
    log_cluster_sizes = np.log(cluster_sizes)
    sensitivities = np.zeros(num_samples)

    denominators = np.ones(num_samples)
    rows = np.arange(num_samples)
    if np.isscalar(R):
        dists = R * euclidean_distances(data, cluster_means)
    elif R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] != num_features:
        raise ValueError('If R is not scalar, it must be a square matrix'
                         ' with the same dimension as the number of features')
    else:
        cholR = np.linalg.cholesky(R)
        dists = euclidean_distances(data.dot(cholR), cluster_means.dot(cholR))
    exp_args = -dists + log_cluster_sizes
    if ball_center is not None:
        data_ball_ip = data.dot(ball_center)[:,np.newaxis]
        data_ball_ip_tiled = np.tile(data_ball_ip, (1, num_clusters))
        center_adjustments = np.abs(data_ball_ip_tiled
                                    - cluster_means.dot(ball_center))
        exp_args -= center_adjustments
        denominators -= np.exp(-dists[rows, cluster_assignments]
                               - center_adjustments[rows, cluster_assignments])
    else:
        denominators -= np.exp(-dists[rows, cluster_assignments])
    denominators += np.sum(np.exp(exp_args), 1)
    return num_samples / denominators


def _calculate_lr_sensitivities(data, R, ball_center, cluster_info):
    """Calculate the sensitivities of each data point for logistic regression.

    This version is very slow, so it should only be used as a reference
    implementation to compare optimized versions against.
    """
    cluster_sizes, cluster_means, cluster_assignments = cluster_info
    num_samples = data.shape[0]
    log_cluster_sizes = np.log(cluster_sizes)
    sensitivities = np.zeros(num_samples)
    for n in xrange(num_samples):
        print(n)
        k = cluster_assignments[n]
        true_mean = cluster_means[k,:].copy()
        # cluster_means[k,:] = cluster_means[k,:]*K/(K-1) - data[n,:]/(K-1),
        # where K = cluster_sizes[k]
        if sp.issparse(data):
            data_row = data[n,:].toarray()
        else:
            data_row = data[n,:]
        K = cluster_sizes[k]
        cluster_means[k,:] *= K / (K - 1.0)
        cluster_means[k,:] -= data_row.squeeze() / (K - 1.0)
        dists = euclidean_distances(data_row.reshape(1,-1), cluster_means)
        dists = dists.squeeze()

        exp_arg = log_cluster_sizes - R * dists
        denominator = 1.0
        if ball_center is not None:
            center_adjustment = np.abs(data_row.dot(ball_center)
                                        - cluster_means.dot(ball_center))
            exp_arg -= center_adjustment
            denominator -= np.exp(-R * dists[k] - center_adjustment[k])
        else:
            denominator -= np.exp(-R * dists[k])
        denominator += np.sum(np.exp(exp_arg))

        sensitivities[n] = num_samples / denominator
        cluster_means[k,:] = true_mean

    return sensitivities


def _sample_indices(num, probs, method=None):
    if method is None or method == 'multinomial':
        return npr.choice(probs.shape[0], size=num, p=probs), 1.0
    else:
        raise ValueError("invalid sampling method '%s'" % method)


def construct_lr_coreset(data, cluster_centers, cluster_assignments=None,
                         R=1.0, ball_center=None, inertia=None,
                         cluster_sizes=None, output_size_param=10000,
                         fast=False, verbose=False):
    """Construct a logistic regression coreset.

    The coreset is contructed for parameters within a ball of radius `R`
    centered at `ball_center`.  If `output_size_param` is an int, outputs a
    coreset with that many data  points. If `output_size_param` is a float
    between 0.0 and 0.5, outputs a  coreset with
    ``int(output_size_param * n_samples + 1)`` data points. If
    ``output_size_param = (eps, delta)``, then probability at least 1-`delta`
    an `eps`-coreset for `data` is ouput.

    Parameters
    ----------
    data : array-like matrix, shape=(n_samples, n_features)

    cluster_centers : array-like matrix, shape=(n_clusters, n_features)
        The clustering specified by `cluster_centers` is used to calculate the
        sensitivity of each data point.

    cluster_assignments : int array-like, shape=(n_samples,), optional
        Assignments of each data point to a cluster. If provided, then
        cluster means and assignments will not be calculated.

    R : float, optional
        Default is 1.0. If negative, adapt the radius to the data.

    ball_center : array-like, shape=(n_features,)
        Default is None, which is equivalent to the centering at the origin.

    inertia : float
        Value of the k-means objective for the proveided clustering of the data.

    cluster_sizes : int array-like, shape=(n_clusters,)

    output_size_param : int or float or tuple of floats, optional
        Default is 10000.

    fast : boolean, optional
        Use fast, approximate LR sensitivity calculation algorithm.
        Default is True.

    verbose : boolean, optional
        Default is False.

    Returns
    -------
    coreset : float ndarray with shape (coreset_size, n_features)

    weights : float ndarray with shape (coreset_size,)

    sensitivities : float ndarray with shape (n_samples,)

    index : int ndarray with shape (coreset_size,)

    success : boolean
        True if a coreset is created. False if the algorithm is unable to
        create a coreset that is substantially smaller than the original data,
        in which case `coreset` is equal to `data` and all the weights are one.
    """
    # Make sure shapes match up and extract sizes
    assert len(data.shape) == 2
    cluster_centers = np.atleast_2d(cluster_centers)
    assert data.shape[1] == cluster_centers.shape[1]
    if cluster_assignments is not None:
        assert len(cluster_assignments.shape) == 1
        assert cluster_assignments.shape[0] == data.shape[0]

    num_samples, num_features = data.shape
    data = as_float_array(data, copy=False)

    # Validate parameters
    max_coreset_size = int(num_samples * MAX_CORESET_PROPORTION + 1)
    if isinstance(output_size_param, int):
        if output_size_param <= 0:
            raise ValueError("If output_size_param is an int, it must be "
                             "positive, but a value of %d was passed" %
                             output_size_param)
        coreset_size = output_size_param
    elif isinstance(output_size_param, float):
        if (output_size_param <= 0.0 or
            output_size_param > MAX_CORESET_PROPORTION):
            raise ValueError("If output_size_param is a float, it must be "
                             "greater than 0.0 and at most %.1f, but a value "
                             "of %.2f was passed" %
                             (MAX_CORESET_PROPORTION, output_size_param))
        coreset_size = max(1, int(num_samples * output_size_param + .5))
    elif isinstance(output_size_param, tuple):
        if (len(output_size_param) != 2 or
            not isinstance(output_size_param[0], float) or
            not isinstance(output_size_param[1], float) or
            output_size_param[0] <= 0.0 or output_size_param[0] > 0.25 or
            output_size_param <= 0.0 or output_size_param[1] >= 1.0):
            raise ValueError("If output_size_param is a tuple, it must be of "
                             "the form (eps, delta) with 0.0 < eps < 0.25 and "
                             "0.0 < delta < 1.0, but a value of %s was passed" %
                             (output_size_param,))
        eps, delta = output_size_param
        coreset_size = None
    else:
        raise ValueError("'output_size_param' must be an int, float, or pair "
                         "of floats")

    # If the coreset size isn't much smaller than the original data size,
    # just return the data
    if coreset_size > max_coreset_size:
        if verbose:
            print("Coreset size would be %d, but only %d samples total" %
                  (coreset_size, num_samples))
        return data, np.ones(num_samples), np.nan, False

    # Calculate cluster info and sensitivities
    if cluster_assignments is None or cluster_sizes is None:
        cluster_info = _calculate_cluster_information(data, cluster_centers,
                                                      cluster_assignments)
    else:
        cluster_info = (cluster_sizes, cluster_centers, cluster_assignments)

    if np.isscalar(R) and R <= 0:
        if inertia is None:
            raise ValueError('Must provide inertia to adaptive R using '
                             'k-means objective')
        if verbose:
            print('Adapting R using k-means objective')
        if R == 0:
            R = -3.0
        normalizer = np.sqrt(inertia / num_samples)
        target_norm = -R
        if ball_center is not None:
            center_norm = la.norm(ball_center)
            if center_norm < target_norm:
                target_norm -= center_norm
        R = target_norm / normalizer
        if verbose:
            print("Normalizer = %.3f" % normalizer)
            print("Adaptive radius = %.3f (target norm = %.3f)" % (R, target_norm))

    if fast:
        sensitivities = _calculate_lr_sensitivities_fast(data, R, ball_center,
                                                         cluster_info)
    else:
        sensitivities = np.zeros(num_samples)
        if sp.issparse(data):
            _algorithms._assign_lr_sensitivities_csr(data, R, ball_center,
                                                     cluster_info[0],
                                                     cluster_info[1],
                                                     cluster_info[2],
                                                     sensitivities)
        else:
            _algorithms._assign_lr_sensitivities_array(data, R, ball_center,
                                                       cluster_info[0],
                                                       cluster_info[1],
                                                       cluster_info[2],
                                                       sensitivities)
    total_sensitivity = np.sum(sensitivities)
    mean_sensitivity = total_sensitivity / num_samples

    # If choosing coreset size adaptively, calculate it and check that it's
    # substantially smaller than the original data size
    if coreset_size is None:
        coreset_size = int(4 * mean_sensitivity / eps**2
                             * ((num_features + 1) * np.log(mean_sensitivity)
                                - np.log(delta)) + 1)
        if coreset_size > max_coreset_size:
            if verbose:
                print("Coreset size would be %d, but only %d samples total "
                      "(mean sensitivity = %.4f)" %
                      (coreset_size, num_samples, mean_sensitivity))
            return data, np.ones(num_samples), sensitivities, False

    # Sample coreset data and calculate weights
    probabilities = sensitivities / total_sensitivity
    coreset_indices, index_weights = _sample_indices(coreset_size,
                                                     probabilities)
    weights = index_weights / probabilities[coreset_indices] / coreset_size
    return data[coreset_indices,:].copy(), weights, sensitivities, True, coreset_indices


def construct_lr_coreset_with_kmeans(data, K, n_init=1, max_iter=1,
                                     R=1.0,  kmeans_subsample_size=None,
                                     output_size_param=10000, fast=True,
                                     verbose=False, 
                                     return_sensitivities=False, index_only=False):
    """Construct LR Coreset, using k-means++ to construct the clustering.

    Parameters
    ----------
    data : array-like matrix, shape=(n_samples, n_features)

    K : int, number of clusters to use in kmeans

    n_init : int, number of kmeans initializations to use

    max_iter : maximum number of kmeans iterations

    R : float, optional
        Default is 1.0.

    kmeans_subsample_size : int or 'auto', optional
        If provided, use a subsample of the data to calculate a clustering.
        If 'auto', then subsample has size min(1000*K, 0.025*n_samples).
        Default is None.

    output_size_param : int or float or tuple of floats, optional
        Default is 10000.

    fast : boolean, optional
        Use fast, approximate LR sensitivity calculation algorithm.
        Default is True.

    verbose : boolean, optional
        Default is False.

    return_sensitivities : boolean, optional
        Default is False.

    Returns
    -------
    coreset : float ndarray with shape (coreset_size, n_features)

    weights : float ndarray with shape (coreset_size,)

    sensitivities : float ndarray with shape (n_samples,)
        Only returned if `return_sensitivities` is True.
    """
    num_samples = data.shape[0]
    if kmeans_subsample_size is None:
        kmeans_subsample_size = num_samples
    if kmeans_subsample_size == 'auto':
        kmeans_subsample_size = int(min(K*1000, num_samples*0.025))
    cluster_sizes = None

    if kmeans_subsample_size < num_samples:
        full_data = data
        samp_indices = npr.choice(num_samples, size=kmeans_subsample_size,
                                  replace=False)
        data = full_data[samp_indices,:]
    if K == 1:
        clusters = np.mean(data, axis=0).reshape(1, -1)
        assignments = np.zeros(data.shape[0])
        inertia = np.sum(euclidean_distances(clusters, data, squared=True))
    else:
        clusters, assignments, inertia = k_means(data, K, n_init=n_init,
                                                 max_iter=max_iter)
    if kmeans_subsample_size < num_samples:
        data = full_data
        cluster_sizes, assignments, inertia = _calculate_assignments(
                                                data, clusters)

    coreset, weights, sensitivities, success, indexes = \
        construct_lr_coreset(data, clusters, cluster_assignments=assignments,
                             R=R, inertia=inertia, cluster_sizes=cluster_sizes,
                             output_size_param=output_size_param, fast=fast)

    if index_only:
        return indexes

    if return_sensitivities:
        return coreset, weights, sensitivities, indexes
    else:
        return coreset, weights, indexes


def random_data_subset(data, output_size_param=10000, max_dim=0):
    """Subsample data without replacement.

    Parameters
    ----------
    data : array-like matrix, shape=(n_samples, n_features)

    output_size_param : int or float, optional
        Default is 10000.

    max_dim : int
        If positive, maximum number of features to use. If zero or negative,
        all features are used. Default is 0.

    Returns
    -------
    subsample : float ndarray with shape (subsample_size, n_features)

    weight : float
    """
    assert len(data.shape) == 2

    num_samples, num_features = data.shape

    if isinstance(output_size_param, int):
        if output_size_param <= 0 or output_size_param > num_samples:
            raise ValueError("If output_size_param is an int, it must be "
                             "positive and no greater than the data sample "
                             "size, but a value of %d was passed" %
                             output_size_param)
        subset_size = output_size_param
    elif isinstance(output_size_param, float):
        if output_size_param <= 0.0 or output_size_param > 1.0:
            raise ValueError("If output_size_param is a float, it must be "
                             "greater than 0.0 and at most 1.0, but a value "
                             "of %.2f was passed" %
                             output_size_param)
        subset_size = max(1, int(num_samples * output_size_param + .5))
    else:
        raise ValueError("'output_size_param' must be an int or float")
    if not isinstance(max_dim, int):
        raise ValueError("'max_dim' must be an int")

    ss_data = data
    if subset_size < num_samples:
        samp_indices = npr.choice(num_samples, size=subset_size, replace=False)
        ss_data = ss_data[samp_indices,:]
    if max_dim > 0 and max_dim < num_features:
        feat_indices = npr.choice(num_features, size=max_dim, replace=False)
        ss_data = ss_data[:,feat_indices]
    weight = float(num_samples) / subset_size
    return ss_data.copy(), weight


def full_data(data):
    """A copy of the full dataset.

    Parameters
    ----------
    data : array-like matrix, shape=(n_samples, n_features)

    output_size_param : int or float, optional
        Default is 10000.

    max_dim : int
        If positive, maximum number of features to use. If zero or negative,
        all features are used. Default is 0.

    Returns
    -------
    data : array-like matrix, with shape (n_samples, n_features)
        A copy of the data

    weight : float
        Always equal to 1.0.
    """
    return data.copy(), 1.0
