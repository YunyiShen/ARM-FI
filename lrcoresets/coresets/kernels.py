# See:
# http://jmlr.org/proceedings/papers/v32/sejdinovic14.pdf
# https://papers.nips.cc/paper/3110-a-kernel-method-for-the-two-sample-problem.pdf
# https://github.com/karlnapf/kameleon-mcmc/blob/master/kameleon_mcmc/kernel/PolynomialKernel.py

from abc import abstractmethod

import numpy as np
import numpy.random as npr
from numpy.lib.index_tricks import fill_diagonal


class Kernel(object):
    """
    Abstract kernel class.
    """
    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__ + "=[]"

    @abstractmethod
    def kernel(self, X, Y=None):
        """
        Calculate the kernel matrix between two samples.

        If `Y` is not provided, then `X` is used in its place.

        Parameters
        ----------
        X : array-like matrix, shape=(n_samplesX, n_features)

        Y : array-like matrix, shape=(n_samplesY, n_features)

        Returns
        -------
        kernel matrix : float ndarray with shape (n_samplesX, n_samplesY)
        """
        raise NotImplementedError()

    # based on kameleon_mcmc implementation
    def estimate_mmd(self, sample1, sample2, unbiased=False):
        """
        Compute the MMD between two samples.

        Parameters
        ----------
        sample1 : array-like matrix, shape=(n_samples1, n_features)

        sample2 : array-like matrix, shape=(n_samples2, n_features)

        unbiased : boolean, optional
            Default is False.

        Returns
        -------
        estimated_mmd : float
        """
        assert len(sample1.shape) == 2
        assert len(sample2.shape) == 2
        assert sample1.shape[1] == sample2.shape[1]

        K11 = self.kernel(sample1, sample1)
        K22 = self.kernel(sample2, sample2)
        K12 = self.kernel(sample1, sample2)
        if unbiased:
            fill_diagonal(K11, 0.0)
            fill_diagonal(K22, 0.0)
            n = float(shape(K11)[0])
            m = float(shape(K22)[0])
            return (np.sum(K11) / (n**2 - n)
                    + np.sum(K22) / (m**2 - m)
                    - 2 * np.mean(K12))
        else:
            return np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)

    def _mmd_from_joint_matrix(self, K, num_samples1):
        return (np.mean(K[:num_samples1,:num_samples1])
                + np.mean(K[num_samples1:,num_samples1:])
                - 2 * np.mean(K[num_samples1:,:num_samples1]))

    # based on kameleon_mcmc implementation
    def two_sample_test(self, sample1, sample2, num_shuffles=1000):
        """
        Compute the p-value associated to the MMD between two samples.
        Uses a standard permutation test to approximate the null.

        Parameters
        ----------
        sample1 : array-like matrix, shape=(n_samples1, n_features)

        sample2 : array-like matrix, shape=(n_samples2, n_features)

        num_shuffles : int, optional
            Default is 1000.

        Returns
        -------
        p_value : float
        """
        num_samples1 = sample1.shape[0]
        num_samples2 = sample2.shape[0]
        merged = np.concatenate([sample1, sample2], axis=0)
        merged_len = merged.shape[0]

        K = self.kernel(merged)
        mmd = self._mmd_from_joint_matrix(K, num_samples1)
        null_samples = np.zeros(num_shuffles)
        for i in range(num_shuffles):
            pp = npr.permutation(merged_len)
            Kpp = K[pp,:][:,pp]
            null_samples[i] = self._mmd_from_joint_matrix(Kpp, num_samples1)

        return np.sum(mmd < null_samples) / float(num_shuffles)


class PolynomialKernel(Kernel):
    """
    The polynomial kernel

    The polynomial kernel is ``k(x,y) = (1 + theta * <x,y>)^degree``.
    """
    def __init__(self, degree, theta=0.2):
        Kernel.__init__(self)
        self.degree = degree
        self.theta = theta

    def __str__(self):
        s = self.__class__.__name__+ "=["
        s += "degree=" + str(self.degree)
        s += "]"
        return s

    def kernel(self, X, Y=None):
        if Y is None:
            Y = X
        return (self.theta + X.dot(Y.T))**self.degree
