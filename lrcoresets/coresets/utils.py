# Authors: Trevor Campbell <tdjc@mit.edu>
#          Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import os
import errno
import inspect

import numpy as np
import scipy.sparse as sp


def create_folder_if_not_exist(path):
    # create the output folder if it doesn't exist
    try:
        os.makedirs(path)
        print('Created output folder:', path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Unknown error creating output directory', path)
            raise


def call_with_superset_args(f, args):
    fargs = inspect.getargspec(f)[0]
    return f(**{k : args[k] for k in args.keys() if k in fargs})


def pretty_file_string_from_dict(d):
    if len(d) == 0:
        return ''
    keys = d.keys()
    keys.sort()
    s = '-'.join(['%s=%s' % (k, d[k]) for k in keys if not callable(d[k])])
    return s


def ensure_dimension_matches(A, B, axis=1):
    if type(A) != type(B):
        raise ValueError('matrices must have same type')
    if axis not in [0,1]:
        raise ValueError("'axis' must be 0 or 1")
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("'A' and 'B' must be two-dimensional")

    A_dim, B_dim = A.shape[axis], B.shape[axis]
    if A_dim == B_dim:
        return A, B
    elif A_dim > B_dim:
        X, Y = B, A
        X_dim, Y_dim = B_dim, A_dim
    else:
        X, Y = A, B
        X_dim, Y_dim = A_dim, B_dim
    if axis == 0:
        extra_shape = (Y_dim - X_dim, X.shape[1])
        if sp.issparse(X):
            stack_fun = sp.vstack
            extra = sp.csr_matrix(extra_shape)
        else:
            stack_fun = np.vstack
            extra = np.zeros(extra_shape)
    else:
        extra_shape = (X.shape[0], Y_dim - X_dim)
        if sp.issparse(X):
            stack_fun = sp.hstack
            extra = sp.csr_matrix(extra_shape)
        else:
            stack_fun = np.hstack
            extra = np.zeros(extra_shape)
    X = stack_fun([X, extra])
    if A_dim > B_dim:
        return Y, X
    else:
        return X, Y
