# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from __future__ import print_function
import sys
from libc.math cimport sqrt, exp, log
from cpython cimport bool
import numpy as np
import scipy.sparse as sp
from scipy.linalg.cython_blas cimport ddot
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()

# inspired by the implementation of
# sklearn.cluster._k_means._assign_labels_array
def _assign_lr_sensitivities_array(np.ndarray[DOUBLE, ndim=2] data,
                                   DOUBLE R,
                                   np.ndarray[DOUBLE, ndim=1] ball_center,
                                   np.ndarray[INT, ndim=1] cluster_sizes,
                                   np.ndarray[DOUBLE, ndim=2] cluster_means,
                                   np.ndarray[INT, ndim=1] cluster_assignments,
                                   np.ndarray[DOUBLE, ndim=1] sensitivities):
    """Calculate the sensitivities of each data point for logistic regression.
    """
    cdef:
        unsigned int num_samples = data.shape[0]
        int num_features = data.shape[1]
        int num_clusters = cluster_sizes.shape[0]
        int data_stride = data.strides[1] / sizeof(DOUBLE)
        int cluster_stride = cluster_means.strides[1] / sizeof(DOUBLE)
        int center_stride = ball_center.strides[0] / sizeof(DOUBLE) if ball_center is not None else 0
        unsigned int sample_idx, cluster_idx
        unsigned int assignment
        np.ndarray[DOUBLE, ndim=1] log_cluster_sizes = np.log(cluster_sizes)
        DOUBLE denominator
        DOUBLE dist
        DOUBLE mean_adjustment
        DOUBLE dat_adjustment
        DOUBLE center_adjustment
        np.ndarray[DOUBLE, ndim=1] cluster_squared_norms = np.zeros(
              num_clusters, dtype=np.float64)
        bool recenter = ball_center is not None
        np.ndarray[DOUBLE, ndim=1] cluster_ball_center_inner_prod = np.zeros(
              num_clusters, dtype=np.float64)
    for cluster_idx in range(num_clusters):
      cluster_squared_norms[cluster_idx] = ddot(
        &num_features, &cluster_means[cluster_idx, 0], &cluster_stride,
        &cluster_means[cluster_idx, 0], &cluster_stride)
      if recenter:
        cluster_ball_center_inner_prod[cluster_idx] = ddot(
            &num_features, &cluster_means[cluster_idx, 0], &cluster_stride,
            &ball_center[0], &center_stride)

    for sample_idx in range(num_samples):
      # print(sample_idx)
      assignment = cluster_assignments[sample_idx]
      # cluster_means[k,:] = cluster_means[k,:]*K/(K-1) - data[n,:]/(K-1),
      # where K = cluster_sizes[k]
      denominator = 1.0
      for cluster_idx in range(num_clusters):
        if cluster_idx == assignment:
          mean_adjustment = cluster_sizes[assignment] / (cluster_sizes[assignment] - 1.0)
          dat_adjustment = 1.0 + 1.0 / (cluster_sizes[assignment] - 1.0)
        else:
          mean_adjustment = 1.0
          dat_adjustment = 1.0
        if recenter:
          datapoint_ball_center_inner_prod = ddot(
              &num_features, &data[sample_idx, 0], &data_stride,
              &ball_center[0], &center_stride)
        dist = 0.0
        dist += (mean_adjustment * dat_adjustment *
                 ddot(&num_features, &data[sample_idx, 0], &data_stride,
                      &cluster_means[cluster_idx, 0], &cluster_stride))
        dist *= -2
        dist += mean_adjustment**2 * cluster_squared_norms[cluster_idx]
        dist += dat_adjustment**2 * ddot(
                    &num_features, &data[sample_idx, 0], &data_stride,
                    &data[sample_idx, 0], &data_stride)
        if dist < 0.0:
          # necessary due to numerical issues for clusters with, for example,
          # all data equal
          dist = 0.0
        else:
          dist = sqrt(dist)
        exp_arg = log_cluster_sizes[cluster_idx] - R * dist
        # print("  1.", dist, exp_arg)
        if recenter:
          center_adjustment = np.abs(
              mean_adjustment * cluster_ball_center_inner_prod[cluster_idx]
            - dat_adjustment * datapoint_ball_center_inner_prod)
          exp_arg -= center_adjustment
        # print("  2.", exp_arg)
        denominator += exp(exp_arg)
        if cluster_idx == assignment:
          if recenter:
            denominator -= np.exp(-R * dist - center_adjustment)
          else:
            denominator -= np.exp(-R * dist)
      sensitivities[sample_idx] = num_samples / denominator

# inspired by the implementation of
# sklearn.cluster._k_means._assign_labels_csr
def _assign_lr_sensitivities_csr(data, DOUBLE R,
                                 np.ndarray[DOUBLE, ndim=1] ball_center,
                                 np.ndarray[INT, ndim=1] cluster_sizes,
                                 np.ndarray[DOUBLE, ndim=2] cluster_means,
                                 np.ndarray[INT, ndim=1] cluster_assignments,
                                 np.ndarray[DOUBLE, ndim=1] sensitivities):
    """Calculate the sensitivities of each data point for logistic regression.
    """
    cdef:
      np.ndarray[DOUBLE, ndim=1] dat_data = data.data
      np.ndarray[INT, ndim=1] dat_indices = data.indices
      np.ndarray[INT, ndim=1] dat_indptr = data.indptr
      unsigned int num_samples = data.shape[0]
      int num_features = data.shape[1]
      int num_clusters = cluster_sizes.shape[0]
      int cluster_stride = cluster_means.strides[1] / sizeof(DOUBLE)
      int center_stride = ball_center.strides[0] / sizeof(DOUBLE) if ball_center is not None else 0
      unsigned int sample_idx, cluster_idx, feature_idx
      unsigned int assignment
      np.ndarray[DOUBLE, ndim=1] log_cluster_sizes = np.log(cluster_sizes)
      DOUBLE denominator
      DOUBLE dist
      DOUBLE datapoint_squared_norm
      DOUBLE datapoint_ball_center_inner_prod
      DOUBLE mean_adjustment
      DOUBLE dat_adjustment
      np.ndarray[DOUBLE, ndim=1] cluster_squared_norms = np.zeros(
            num_clusters, dtype=np.float64)
      bool recenter = ball_center is not None
      np.ndarray[DOUBLE, ndim=1] cluster_ball_center_inner_prod = np.zeros(
            num_clusters, dtype=np.float64)

    for cluster_idx in range(num_clusters):
      cluster_squared_norms[cluster_idx] = ddot(
          &num_features, &cluster_means[cluster_idx, 0], &cluster_stride,
          &cluster_means[cluster_idx, 0], &cluster_stride)
      if recenter:
        cluster_ball_center_inner_prod[cluster_idx] = ddot(
            &num_features, &cluster_means[cluster_idx, 0], &cluster_stride,
            &ball_center[0], &center_stride)

    for sample_idx in range(num_samples):
      assignment = cluster_assignments[sample_idx]
      # cluster_means[k,:] = cluster_means[k,:]*K/(K-1) - data[n,:]/(K-1),
      # where K = cluster_sizes[k]
      denominator = 1.0
      for cluster_idx in range(num_clusters):
        if cluster_idx == assignment:
          mean_adjustment = cluster_sizes[assignment] / (cluster_sizes[assignment] - 1.0)
          dat_adjustment = 1.0 + 1.0 / (cluster_sizes[assignment] - 1.0)
        else:
          mean_adjustment = 1.0
          dat_adjustment = 1.0
        dist = 0.0
        datapoint_squared_norm = 0.0
        datapoint_ball_center_inner_prod = 0.0
        for k in range(dat_indptr[sample_idx], dat_indptr[sample_idx + 1]):
          dist += cluster_means[cluster_idx, dat_indices[k]] * dat_data[k]
          datapoint_squared_norm += dat_data[k] * dat_data[k]
          if recenter:
            datapoint_ball_center_inner_prod += dat_data[k] * ball_center[k]
        dist *= -2 * mean_adjustment * dat_adjustment
        dist += mean_adjustment**2 * cluster_squared_norms[cluster_idx]
        dist += dat_adjustment**2 * datapoint_squared_norm
        if dist < 0.0:
          # necessary due to numerical issues for clusters with, for example,
          # all data equal
          dist = 0.0
        else:
          dist = sqrt(dist)
        exp_arg = log_cluster_sizes[cluster_idx] - R * dist
        if recenter:
          exp_arg -= np.abs(
              mean_adjustment * cluster_ball_center_inner_prod[cluster_idx]
            - dat_adjustment * datapoint_ball_center_inner_prod)
        denominator += exp(exp_arg)
        if cluster_idx == assignment:
          denominator -= exp(-R * dist)
      sensitivities[sample_idx] = num_samples / denominator
