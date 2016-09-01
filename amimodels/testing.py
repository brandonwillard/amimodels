import numpy as np
import pymc

from amimodels.normal_hmm import HMMStateSeq, TransProbMatrix
from amimodels.deterministics import HMMLinearCombination


def assert_hpd(stochastic, true_value, alpha=0.05,
               subset=slice(0, None), rtol=0):
    r""" Assert that the given stochastic's :math:`(1-\alpha)` HPD
    interval covers the true value.

    Parameters
    ==========
    stochastic: a pymc.Stochastic
        The stochastic we want to test.  Must have trace values.
    true_value: ndarray
        The "true" values to check against.
    alpha: float, optional
        The alpha confidence level.
    subset: slice, optional
        Slice for the subset of parameters to check.
    rtol: array of float, optional
        Relative tolerences for matching the edges of the intervals.
    edge

    Returns
    =======
        Nothing, just exec's the assert statements.
    """
    interval_name = '{}% HPD interval'.format(int(100. * (1. - alpha)))
    stoc_stats = stochastic.stats(alpha=alpha)[interval_name]

    true_value = np.ravel(true_value)[subset]

    lower_bounds = np.ravel(stoc_stats[0])[subset]
    upper_bounds = np.ravel(stoc_stats[1])[subset]

    if np.alen(rtol) == 1:
        rtol = np.tile(rtol, np.alen(true_value))

    lower_diff = lower_bounds - true_value
    lower_ok = lower_diff <= 0
    lower_close = np.allclose(lower_bounds[~lower_ok], true_value[~lower_ok],
                              atol=1e-5, rtol=rtol[~lower_ok])
    lower_check = lower_close

    assert lower_check, "{} lower 95% HPD interval".format(stochastic.__name__)

    upper_diff = true_value - upper_bounds
    upper_ok = upper_diff <= 0
    upper_close = np.allclose(upper_bounds[~upper_ok], true_value[~upper_ok],
                              atol=1e-5, rtol=rtol[~upper_ok])
    upper_check = upper_close

    assert upper_check, "{} upper 95% HPD interval".format(stochastic.__name__)


def simple_norm_reg_model(N_obs=100, X_matrices=None, betas=None,
                          trans_mat_obs=np.asarray([[0.9], [.1]]),
                          tau_y=100, y_obs=None):
    r""" A simple normal observations/emissions HMM regression model with fixed
    transition matrix.  Uses `HMMStateSeq` and `HMMLinearCombination` to model
    the HMM states and observation mean, respectively.

    This is useful for generating simple test/toy data according to a
    completely known model and parameters.

    Parameters
    ==========
    N_obs: int
        Number of observations.
    X_matrices: list of ndarray or None
        List of design matrices for the regression parameters.  If `None`,
        design matrices are generated for the given `betas` (which themselves
        will be generated if `None`).
    betas: list of ndarray or pymc.Stochastic
        List of regression parameters matching X_matrices.  If `None`, these
        are randomly generated according to the shapes of `X_matrices`.
    trans_mat_obs: ndarray or pymc.Stochastic
        Transition probability matrix.
    tau_y: float or ndarray
        Observation variance.
    y_obs: ndarray
        A vector of observations.  If `None`, a non-observed pymc.Stochastic is
        produced.


    Returns
    =======
        A dictionary containing all the variables in the model.
    """

    if betas is None:
        if X_matrices is None:
            betas = [np.random.randn(2), np.random.randn(2)]
            X_matrices = [np.random.randn(N_obs, 2),
                          np.random.randn(N_obs, 2)]
        else:
            betas = [np.random.randn(X_.shape[1]) for X_ in X_matrices]

    if X_matrices is None:
        X_matrices = [np.random.randn(N_obs, np.alen(b_)) for b_ in betas]

    states_rv = HMMStateSeq("states", trans_mat_obs, N_obs)

    mu = HMMLinearCombination('mu', X_matrices, betas, states_rv)

    y_rv = pymc.Normal('y', mu, tau_y, value=y_obs,
                       observed=(y_obs is not None))

    return locals()


def simple_state_seq_model(mu_vals=np.asarray([-1, 1]),
                           trans_mat_obs=np.asarray([[0.9], [.1]]),
                           N_obs=100, tau_y=100, y_obs=None):
    r""" A simple normal observations/emissions HMM model with fixed transition
    matrix.  Uses `HMMStateSeq` and an array-indexing deterministic for the
    observations' means.

    This is useful for generating simple test/toy data according to a
    completely known model and parameters.

    Parameters
    ==========
    mu_vals: list of ndarray or pymc.Stochastic
        List of mean values for the observations/emissions of each
        state.
    trans_mat_obs: ndarray or pymc.Stochastic
        Transition probability matrix.
    N_obs: int
        Number of observations.
    tau_y: float or ndarray
        Observation variance.
    y_obs: ndarray
        A vector of observations.  If `None`, a non-observed pymc.Stochastic is
        produced.


    Returns
    =======
        A dictionary containing all the variables in the model.
    """

    states_rv = HMMStateSeq("states", trans_mat_obs, N_obs)

    @pymc.deterministic(trace=True, plot=False)
    def mu(states_=states_rv):
        return mu_vals[states_]

    y_rv = pymc.Normal('y', mu, tau_y, value=y_obs,
                       observed=(y_obs is not None))

    return locals()


def simple_state_trans_model(mu_vals=np.asarray([-1, 1]),
                             alpha_trans=np.asarray([[1., 10.], [10., 1.]]),
                             N_obs=100, tau_y=100, y_obs=None):
    r""" A simple normal observations/emissions HMM model with a stochastic
    transition matrix.  Uses `HMMStateSeq`, `TransProbMatrix` and an
    array-indexing deterministic for the observations' means.

    This is useful for generating simple test/toy data according to a
    completely known model and parameters.

    Parameters
    ==========
    mu_vals: list of ndarray or pymc.Stochastic
        List of mean values for the observations/emissions of each
        state.
    alpha_trans: ndarray
        Dirichlet hyper-parameters for the transition probability matrix's
        prior.
    N_obs: int
        Number of observations.
    tau_y: float or ndarray
        Observation variance.
    y_obs: ndarray
        A vector of observations.  If `None`, a non-observed pymc.Stochastic is
        produced.


    Returns
    =======
        A dictionary containing all the variables in the model.
    """

    trans_mat_rv = TransProbMatrix('trans_mat', alpha_trans)
    states_rv = HMMStateSeq("states", trans_mat_rv, N_obs)

    @pymc.deterministic(trace=True, plot=False)
    def mu(states_=states_rv):
        return mu_vals[states_]

    y_rv = pymc.Normal('y', mu, tau_y, value=y_obs,
                       observed=(y_obs is not None))

    return locals()
