# Author: Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import numpy as np
import numpy.random as npr


def gelman_rubin_diagonostic(samples, multivariate=False, norm_ord=np.inf):
    """Calculate the Gelman-Rubin statistic.

    Gelman, Andrew, and Donald B. Rubin. "Inference from iterative simulation
    using multiple sequences." Statistical Science (1992): 457-472.

    Brooks, Stephen P., and Andrew Gelman. "General methods for monitoring
    convergence of iterative simulations." Journal of Computational and
    Graphical Statistics 7.4 (1998): 434-455.

    Parameters
    ----------
    samples : array-like, shape=(n_chains, n_samples, n_features)

    multivariate : boolean, optional
        Flag indicating that the multivariate version of the diagnostic as
        defined by Brooks and Gelman should be used. Default is False.

    norm_ord : float, optional
        In the non-multivariate, multidimensional case, norm to use for
        averaging the 1-dimensional statistics. Default is infinity.

    Returns
    -------
    R_hat : float
    """
    assert samples.shape[0] > 1, "Must have samples from more than one chain"
    is_1d = len(samples.shape) == 2
    if not multivariate and not is_1d:
        Rhats = [gelman_rubin_diagonostic(samples[:,:,i])
                    for i in range(samples.shape[2])]
        assert norm_ord > 0.0
        normalizer = samples.shape[2]**(1./norm_ord)
        return np.linalg.norm(Rhats, ord=norm_ord) / normalizer
    num_chains = samples.shape[0]
    chain_means = samples.mean(axis=1)
    B = np.cov(chain_means.T)
    W = 0
    for m in range(num_chains):
        if is_1d:
            W += np.cov(samples[m,:])
        else:
            W += np.cov(samples[m,:,:].T)
    W /= num_chains
    adj = (samples.shape[1] - 1.) / samples.shape[1]
    if is_1d:
        lambda_1 = B / W
    else:
        WinvB = np.linalg.solve(W, B)
        lambda_1 = np.linalg.norm(WinvB, ord=2)
    #lambda_1 = linalg.eigh(WinvB, eigvals_only=True, eigvals=())[0]
    R_hat = adj + (1. + 1. / num_chains) * lambda_1
    return R_hat


def _ensure_positive_int(val, name):
    if not isinstance(val, int) or val <= 0:
        raise ValueError("'%s' must be a positive integer")
    return True


def _ensure_callable(val, name):
    if not callable(val):
        raise ValueError("'%s' must be a callable")
    return True


def _adapt_param(value, i, log_accept_prob, target_rate, const=3):
    """
    Adapt the value of a parameter.
    """
    new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
    # new_val = max(min_val, min(max_val, new_val))
    return new_val


def mh(x0, p, q, sample_q, steps=1, warmup=None, thin=1,
       proposal_param=None, target_rate=0.234):
    """
    (Adaptive) Metropolis Hastings sampling.

    Parameters
    ----------
    x0 : object
        The initial state.

    p : function
        Accepts one argument `x` and outputs the log probability density of the
        target distribution at `x`.

    q : function or None
        Accepts two arguments, `x` and `xf`. Outputs the log proposal density
        of going from `x` to `xf`. None indicates the proposal is symmetric,
        so there is no need to calculate the proposal probability when
        deciding whether to accept the move to `xf`.

    sample_q : function
        Accepts one argument `x` and proposes `xf` given `x`.

    steps : int, optional
        The number of MH steps to take. Default is 1.

    warmup : int, optional
        The number of warmup (aka burnin) iterations. Default is ``steps/2``.

    thin : int, optional
        Period for saving samples. Default is 1.

    proposal_param : numeric, optional
        If proveided then use adaptive MH targeting an accept rate of
        `target_rate`. In this case `sample_q` and `q` should both accept
        `proposal_param` as an additional final argument. Default is None.

    target_rate : float, optional
        Default is 0.234.

    Returns
    -------
    samples : array with length ``(steps - warmup) / thin``

    accept_rate : float
        Calculated from non-warmup iterations.
    """
    # Validate parameters
    _ensure_callable(p, 'p')
    if q is not None:
        _ensure_callable(q, 'q')
    _ensure_callable(sample_q, 'sample_q')
    _ensure_positive_int(steps, 'steps')
    _ensure_positive_int(thin, 'thin')
    if warmup is None:
        warmup = steps / 2
    else:
        _ensure_positive_int(warmup, 'warmup')
        if warmup >= steps:
            raise ValueError("Number of warmup iterations is %d, which is "
                             "greater than the total number of steps, %d" %
                             (warmup, steps))

    # Run (adaptive) MH algorithm
    accepts = 0.0
    xs = []
    x = x0
    for step in range(steps):
        # Make a proposal
        p0 = p(x)
        if proposal_param is None:
            xf = sample_q(x)
        else:
            xf = sample_q(x, proposal_param)
        pf = p(xf)

        # Compute acceptance ratio and accept or reject
        odds = pf - p0
        if q is not None:
            if proposal_param is None:
                qf, qr = q(x, xf), q(xf, x)
            else:
                qf, qr = q(x, xf, proposal_param), q(xf, x, proposal_param)
            odds += qr - qf
        if proposal_param is not None and step < warmup:
                proposal_param = _adapt_param(proposal_param, step,
                                              min(0, odds), target_rate)
        if np.log(npr.rand()) < odds:
            x = xf
            if step >= warmup:
                accepts += 1

        if step >= warmup and (step - warmup) % thin == 0:
            xs.append(x)

    accept_rate = accepts / (steps - warmup)
    if len(xs) > 1:
        return xs, accept_rate
    else:
        return xs[0], accept_rate
