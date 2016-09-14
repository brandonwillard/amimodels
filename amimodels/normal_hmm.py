r"""
This module provides classes and functions for producing and simulating hidden
Markov models (HMM) with scalar normal/Gaussian-distributed observations in
PyMC.

.. moduleauthor:: Brandon T. Willard
"""
import numpy as np
import scipy
import pandas as pd
import patsy
import pymc

from .stochastics import HMMStateSeq, TransProbMatrix
from .hmm_utils import compute_trans_freqs, compute_steady_state
from .deterministics import (HMMLinearCombination, KIndex, NumpyTake,
                             NumpyChoose, NumpyHstack)


def get_stochs_excluding(stoch, excluding):
    """ Get the parents of a stochastic excluding the given
    list of stochastic and/or parent names.

    Parameters
    ==========
    stoch: pymc.Stochastic
        Root stochastic/node.
    excluding: list of str
        Stochastic/node and parent names to exclude.
    """
    res = set()
    for s_ in stoch.extended_parents:
        if s_.__name__ in excluding or set(s_.parents.keys()) & excluding:
            res |= get_stochs_excluding(s_, excluding)
        else:
            res.add(s_)
    return res


def trace_sampler(model, stoch, traces, dbname=None):
    """ Creates a PyMC ram database for stochastic
    given a model and set of trace values for its parent
    stochastics.

    Parameters
    ==========
    model: pymc.Model object
        The model object.
    stoch: pymc.Stochastic or str
        The stochastic, or name, for which we want values under the
        given samples in `traces`.
    traces: dict of str, numpy.ndarray
        A dictionary of `stoch`'s parents' stochastic names
        and trace values

    Returns
    =======
    A pymc.database.ram.Database.

    """
    if dbname is None:
        dbname = model.__name__

    ram_db = pymc.database.ram.Database(dbname)

    if isinstance(stoch, pymc.Node):
        stoch_name = stoch.__name__
    else:
        stoch_name = stoch

    target_stoch = model.get_node(stoch_name)

    stoch_value_fn = {'mu': target_stoch.get_value}

    stochs_to_set = set(filter(lambda x: x.__name__ in traces.keys(),
                               model.nodes))

    stochs_to_get = target_stoch.extended_parents - stochs_to_set

    for s_ in stochs_to_set | stochs_to_get:
        stoch_value_fn[s_.__name__] = s_.get_value

    mcmc_iters = np.alen(traces.values()[0])
    ram_db._initialize(stoch_value_fn, mcmc_iters)

    for n in xrange(mcmc_iters):
        for s_ in stochs_to_set:
            s_.value = traces[s_.__name__][n]
        for s_ in stochs_to_get:
            _ = s_.random()
        ram_db.tally()

    return ram_db


class NormalHMMInitialParams(object):
    r"""An object that holds initial parameters for a
    normal-observations HMM.
    """

    def __init__(self, alpha_trans, trans_mat, states, betas, Ws, Vs, p0):
        """ Object containing initial values for the normal HMM.

        Parameters
        ==========
        alpha_trans: numpy.array of float
            Initial Dirichlet parameters for the transition probability
            matrix's first K-1 columns.
        trans_mat: numpy.array of float
            A transition probability matrix for K states with shape
            (K, K-1), i.e. the last column omitted.
        states: numpy.array of int
            Initial states vector with values in an integer range
            :math:`[0,\dots,N_S-1]`, where :math:`N_S` is the number of hidden
            states.  Should be length of observations (i.e. `y_obs`).
        betas: numpy.array of float
            Initial value matrix with rows corresponding to the regression
            parameter vectors for each hidden state.  Be aware of any implicit
            ordering; generally assume that the order is lower to higher mean
            value (e.g. :math:`\mu_t^{(1)} < \mu_t^{(2)}` with
            :math:`\mu_t^{(S_t)} = X^{(S_t)}_t^\top \beta^{(S_t)}`).
        Ws: numpy.array of float
            Matrix of initial state regression parameter prior covariances.  We
            currently assume that rows correspond to a vector of independent
            diagonal covariance terms.
        Vs: numpy.array of float
            Vector of initial observation covariances.
        p0: numpy.array of float
            Vector of initial state probabilities.  If `None`, then the steady
            state of the probability transition matrix is used.
        """

        self.alpha_trans = alpha_trans
        self.trans_mat = trans_mat
        self.states = states
        self.betas = betas
        self.Ws = Ws
        self.Vs = Vs
        self.p0 = p0

    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4,
                                                          width=1)


class NormalHMMProcess(object):
    r"""An object that produces simulations from a normal-observations HMM.
    """

    def __init__(self, trans_mat, N_obs, p0, betas, Vs,
                 exogenous_sim_func=None, formulas=None, start_datetime=None,
                 seed=None):
        """
        Parameters
        ==========
        N_obs: int
            Number of observations.
        trans_mat: numpy.array of float
            A transition probability matrix for K states with shape
            (K, K-1), i.e. the last column omitted.
        p0: numpy.array of int
            Initial state probabilities.
        betas: tuple, list of numpy.array
            Matrix of state parameters.  Rows of the matrix
            correspond to the states.
        Vs: numpy.array of float
            Observation variances for each state.
        exogenous_sim_func: function or None
            A function that takes a `pandas.tseries.index.DatetimeIndex` and
            returns a `pandas.DataFrame` with the same index and named
            exogenous variable columns.
            If `None`, the default function produces a `temp` column.
        formulas: tuple, list of str
            `patsy` formula strings used to generate design matrices for
            each state distribution.
            The names used in the formulae should match the ones produced
            by `exogenous_sim_func`.
        start_datetime: pandas.tslib.Timestamp
            Starting datetime.
        seed: int
            Random number generator seed.  If `None` is provided (default),
            one will be automatically created by `numpy.RandomState`.
        """

        self.N_states = len(p0)

        if not (trans_mat.ndim == 2 and
                trans_mat.shape[0] == self.N_states and
                trans_mat.shape[1] == self.N_states - 1):
            raise ValueError()

        if len(betas) != self.N_states:
            raise ValueError()

        if len(Vs) != self.N_states:
            raise ValueError()

        if formulas is not None and len(formulas) != self.N_states:
            raise ValueError()

        self.trans_mat = trans_mat
        self.N_obs = N_obs
        self.p0 = p0
        self.betas = betas
        self.Vs = Vs

        if exogenous_sim_func is None:
            exogenous_sim_func = generate_temperatures

        self.exogenous_sim_func = exogenous_sim_func

        self.start_datetime = start_datetime
        if self.start_datetime is None:
            self.start_datetime = pd.tslib.Timestamp(pd.datetime.now())

        if formulas is None:
            formulas = ["1 ~ 1"] * self.N_states

        self.formulas = formulas

        #
        # P_true[j, i] = p(S_t = i | S_{t-1} = j)
        #
        self.P = np.column_stack((self.trans_mat, 1. -
                                  self.trans_mat.sum(axis=1)))

        # Compute the steady state by solving the characteristic equation.
        Lam = np.eye(self.N_states) - self.P + np.ones((self.N_states,) * 2)
        self.u = np.linalg.solve(Lam.T, np.ones(self.N_states))

        self.rng = np.random.RandomState(seed=seed)

    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4,
                                                          width=1)

    def generate_exogenous(self):
        """ Generate exogenous terms/covariates and time indices.
        Override this if you will, but make sure the dimensions match
        `self.betas`.

        Returns
        =======
        index_sim
            A `pandas.DatetimeIndex`.
        X_matrices
            A sequence of design matrices corresponding to each `self.betas`.
        """
        index_sim = pd.date_range(start=self.start_datetime,
                                  periods=self.N_obs,
                                  freq='H')

        X_data = self.exogenous_sim_func(index_sim)

        X_matrices = tuple()
        for s in range(self.N_states):
            _, X = patsy.dmatrices(self.formulas[s], X_data,
                                   return_type='dataframe')
            X_matrices += (X,)

        return index_sim, X_matrices

    def simulate(self):
        """ Simulate a series from this normal-emissions HMM.

        Returns
        =======
        states: numpy.array of int
            Simulated state values.
        y: pandas.DataFrame
            Time series of simulated usage observations.
        X_matrices: list of pandas.DataFrame
            List of `pandas.DataFrame`s for each `self.betas` with designs
            given by `self.formulas`.
        """

        states = np.empty(self.N_obs, dtype=np.uint8)
        usage_sim = np.empty(self.N_obs, dtype=np.float)

        index_sim, X_matrices = self.generate_exogenous()

        for t in range(self.N_obs):
            p_now = self.p0 if t == 0 else self.P[states[t-1]]
            state_now = int(self.rng.choice(self.N_states, 1, p=p_now))
            states[t] = state_now
            mu_now = np.dot(X_matrices[state_now].iloc[t, :],
                            self.betas[state_now])

            usage_sim[t] = scipy.stats.truncnorm.rvs(0., np.inf,
                                                     loc=mu_now,
                                                     scale=np.sqrt(self.Vs[state_now]),
                                                     random_state=self.rng)

        y = pd.DataFrame(usage_sim, index=index_sim, columns=["usage"])

        return states, y, X_matrices


def generate_temperatures(ind, period=24., offset=0., base_temp=60.,
                          flux_amt=10.):
    """ Generate very regular temperature oscillations.
    This is roughly based on observations starting at 4/1/2015 in CA (in UTC!).

    Parameters
    ==========

    ind: pandas.DatetimeIndex
        Time index over which the temperatures will be computed.
    period: float
        Period of the sinusoid.
    offset: float
        Frequency offset.
    base_temp: float
        Temperature around which the sinusoid will fluctuate.
    flux_amt: float
        Scaling intensity of sinusoid.
    """
    epoch_hours = ind.astype(np.int64) // (10**9 * 60 * 60)
    res = pd.DataFrame(base_temp + flux_amt * np.cos(2. * np.pi * epoch_hours /
                                                     period + offset),
                       index=ind, columns=["temp"])
    return res


def calc_alpha_prior(obs_states, N_states, trans_freqs=None):
    """ A method of producing informed Dirichlet distribution parameters
    from observed states.

    Parameters
    ==========
    obs_states: ndarray of int
        Array of state label observations.
    N_states: int
        Total number of states (max integer label of observations).
    trans_freqs: numpy.array of float
        Empirical transition probabilities for obs_states sequence.

    Returns
    =======
    numpy.array of Dirichlet parameters initialized/updated by the observed
    sequence.
    """
    if trans_freqs is None:
        alpha_trans = compute_trans_freqs(obs_states, N_states)
    else:
        alpha_trans = trans_freqs.copy()
    alpha_trans *= np.sqrt(obs_states.size)
    alpha_trans += 1
    alpha_trans /= alpha_trans.sum(axis=1, keepdims=True)
    alpha_trans *= np.sqrt(obs_states.size)

    return alpha_trans


def gmm_norm_hmm_init_params(y, X_matrices):
    """ Generates initial parameters for the univariate normal-emissions HMM
    with normal mean priors.

    Parameters
    ==========
    y: pandas.DataFrame or pandas.Series
        Time-indexed vector of observations.
    X_matrices: list of pandas.DataFrame
        Collection of design matrices for each hidden state's mean.

    Returns
    =======
    init_params:
        A `NormalHMMInitialParams` object.
    """

    # initialize with simple gaussian mixture
    from sklearn.mixture import GMM

    N_states = len(X_matrices)

    gmm_model = GMM(N_states, covariance_type='diag')
    gmm_model_fit = gmm_model.fit(y.dropna())

    from operator import itemgetter
    gmm_order = sorted(enumerate(gmm_model_fit.means_), key=itemgetter(1))
    gmm_order = map(itemgetter(0), gmm_order)
    gmm_order_map = dict(zip(gmm_order, range(len(gmm_order))))
    # gmm_ord_weights = np.asarray([gmm_model_fit.weights_[x] for x in
    #                               gmm_order])

    # TODO: attempt conditional regression when X matrices tell us
    # that we'll be fitting regression terms?
    # For now we just set those terms to zero.
    gmm_ord_means = np.asarray([np.append(gmm_model_fit.means_[x], [0.] *
                                          (X_matrices[i].shape[1]-1)) for i, x
                                in enumerate(gmm_order)])
    gmm_ord_obs_covars = np.asarray([gmm_model_fit.covars_[x, 0] for x in
                                     gmm_order])
    gmm_states = pd.DataFrame(None, index=y.index, columns=['state'],
                              dtype=np.int)
    gmm_raw_predicted = gmm_model_fit.predict(y.dropna()).astype(np.int)
    gmm_states[~y.isnull().values] = gmm_raw_predicted[:, None]

    from functools import partial
    gmm_lam = partial(lambda x: gmm_order_map.get(x, np.nan))
    gmm_ord_states = gmm_states['state'].map(gmm_lam)

    beta_prior_covars = [np.ones(X_matrices[i].shape[1]) * 10 for i in
                         range(len(X_matrices))]

    trans_freqs = compute_trans_freqs(gmm_ord_states, N_states)
    alpha_trans_0 = calc_alpha_prior(gmm_ord_states, N_states,
                                     trans_freqs=trans_freqs)

    if any(y.isnull()):
        # Now, let's sample values for the missing observations.
        # TODO: Would be better if we did this according to the
        # initial transition probabilities, no?
        for t in np.arange(y.size)[y.isnull().values.ravel()]:
            if t == 0:
                p0 = compute_steady_state(trans_freqs[:, :-1])
                state = pymc.rcategorical(p0)
            else:
                state = pymc.rcategorical(trans_freqs[int(gmm_ord_states[t-1])])
            gmm_ord_states[t] = state

    init_params = NormalHMMInitialParams(alpha_trans_0, None, gmm_ord_states,
                                         gmm_ord_means, beta_prior_covars,
                                         gmm_ord_obs_covars, None)

    return init_params


def bic_norm_hmm_init_params(y, X_matrices):
    """ Initialize a normal HMM regression mixture with a GMM mixture
    of a BIC determined number of states.  Starting with an initial
    set of design matrices, this function searches for the best number
    of additional constant states to add to the model.


    Parameters
    ==========
    y: pandas.DataFrame or pandas.Series
        Time-indexed vector of observations.
    X_matrices: list of pandas.DataFrame
        Collection of design matrices for each initial state.

    Returns
    =======
    init_params:
        A `NormalHMMInitialParams` object.
    """

    N_states = len(X_matrices)

    from sklearn import mixture
    lowest_bic = np.infty
    bic = []
    for n_components in range(N_states, 10):
        gmm = mixture.GMM(n_components=n_components,
                          covariance_type="diag")
        _ = gmm.fit(y.dropna())
        bic.append(gmm.bic(y.dropna()))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    from operator import itemgetter
    gmm_order = sorted(enumerate(best_gmm.means_), key=itemgetter(1))
    gmm_order = np.asarray(map(itemgetter(0), gmm_order))
    gmm_order_map = dict(zip(gmm_order, range(len(gmm_order))))

    gmm_states = pd.DataFrame(None, index=y.index, columns=['state'],
                              dtype=np.int)
    gmm_raw_predicted = best_gmm.predict(y.dropna()).astype(np.int)
    gmm_states[~y.isnull().values.ravel()] = gmm_raw_predicted[:, None]

    from functools import partial
    gmm_lam = partial(lambda x: gmm_order_map.get(x, np.nan))
    states_ordered = gmm_states['state'].map(gmm_lam)

    # When best_gmm.n_components > N_states we need to map multiple
    # GMM estimated states to a single state (the last, really) in
    # the model.  Below we create the map that says which states
    # in GMM map to which states in the model.
    from itertools import izip_longest
    from collections import defaultdict
    gmm_to_state_map = dict(izip_longest(range(best_gmm.n_components),
                                         range(N_states),
                                         fillvalue=N_states-1))
    state_to_gmm_map = defaultdict(list)
    for i, v in zip(gmm_to_state_map.values(), gmm_to_state_map.keys()):
        state_to_gmm_map[i].append(v)

    gmm_to_state_lam = partial(lambda x: gmm_to_state_map.get(x, np.nan))

    states_initial = states_ordered.map(gmm_to_state_lam)

    trans_freqs = compute_trans_freqs(states_initial, N_states)
    alpha_trans_0 = calc_alpha_prior(states_initial, N_states,
                                     trans_freqs=trans_freqs)

    if any(y.isnull()):
        # Now, let's sample values for the missing observations.
        # TODO: Would be better if we did this according to the
        # initial transition probabilities, no?
        for t in np.arange(y.size)[y.isnull().values.ravel()]:
            if t == 0:
                p0 = compute_steady_state(trans_freqs[:, :-1])
                state = pymc.rcategorical(p0)
            else:
                state = pymc.rcategorical(trans_freqs[int(states_initial[t-1])])
            states_initial[t] = state

    beta_prior_means = []
    beta_prior_covars = []
    obs_prior_vars = np.empty(N_states)
    for i, gmm_states in state_to_gmm_map.items():

        this_order = gmm_order[gmm_states]
        these_weights = best_gmm.weights_[this_order]
        these_weights /= these_weights.sum()
        these_means = best_gmm.means_[this_order]
        these_covars = best_gmm.covars_[this_order]

        # Use the exact mixture variance when we have two
        # states to combine; otherwise, use a crude estimate.
        # TODO: We can get an expression for len(gmm_states) > 2.
        if len(gmm_states) == 2:
            pi_, pi_n = these_weights
            sigma_1, sigma_2 = these_covars
            mu_diff = np.ediff1d(these_means)
            this_cov = pi_ * (sigma_1**2 + mu_diff**2 * pi_n**2) +\
                pi_n * (sigma_2**2 + mu_diff**2 * pi_**2)
            this_cov = float(this_cov)
        else:
            this_cov = these_covars.sum()

        # TODO: How should/could we use this?
        # this_mean = np.dot(these_weights, best_gmm.means_[this_order])

        # Get the data conditional on this [estimated] state.
        states_cond = np.asarray(map(lambda x: True if x in gmm_states else
                                     False, states_ordered))
        from sklearn import linear_model
        reg_model = linear_model.ElasticNetCV(fit_intercept=False)

        N_beta = X_matrices[i].shape[1]

        X_cond = X_matrices[i][states_cond]
        y_cond = y[states_cond].get_values().ravel()

        if not X_cond.empty:
            # TODO: Could ask how this compares to the intercept-only model above.
            reg_model_fit = reg_model.fit(X_cond, y_cond)
            reg_model_err = np.atleast_1d(np.var(reg_model_fit.predict(X_cond) -
                                                y_cond))
            beta_prior_means += [np.atleast_1d(reg_model_fit.coef_)]
            beta_prior_covars += [np.repeat(reg_model_err, N_beta)]
        else:
            beta_prior_means += [np.zeros(N_beta)]
            # TODO: A better default for an "uninformed" initial value?
            # This is really a job for a prior distribution.
            beta_prior_covars += [100. * np.ones(N_beta)]

        obs_prior_vars[i] = this_cov

    init_params = NormalHMMInitialParams(alpha_trans_0, None, states_initial,
                                         beta_prior_means, beta_prior_covars,
                                         obs_prior_vars, None)
    return init_params


def make_normal_hmm(y_data, X_data, initial_params=None, single_obs_var=False,
                    include_ppy=False):
    r""" Construct a PyMC2 scalar normal-emmisions HMM model of the form

    .. math::

        y_t &\sim \operatorname{N}^{+}(x_t^{(S_t)\top} \beta^{(S_t)}, V^{(S_t)}) \\
        \beta^{(S_t)}_i &\sim \operatorname{N}(m^{(S_t)}, C^{(S_t)}),
        \quad i \in \{1,\dots, M\} \\
        S_t \mid S_{t-1} &\sim \operatorname{Categorical}(\pi^{(S_{t-1})}) \\
        \pi^{(S_t-1)} &\sim \operatorname{Dirichlet}(\alpha^{(S_{t-1})})

    where :math:`\operatorname{N}_{+}` is the positive (truncated below zero)
    normal distribution, :math:`S_t \in \{1, \ldots, K\}`,
    :math:`C^{(S_t)} = \lambda_i^{(S_t) 2} \tau^{(S_t) 2}` and

    .. math::

        \lambda^{k}_i &\sim \operatorname{Cauchy}^{+}(0, 1) \\
        \tau^{(k)} &\sim \operatorname{Cauchy}^{+}(0, 1) \\
        V^{-(k)} &\sim \operatorname{Gamma}(n_0/2, n_0 S_0/2)

    for :math:`k \in \{1, \ldots, K\}`.


    for observations :math:`y_t` in :math:`t \in \{0, \dots, T\}`,
    features :math:`x_t^{(S_t)} \in \mathbb{R}^M`,
    regression parameters :math:`\beta^{(S_t)}`, state sequences :math:`\{S_t\}^T_{t=1}` and
    state transition probabilities :math:`\pi \in [0, 1]^{K}`.
    :math:`\operatorname{Cauchy}^{+}` is the standard half-Cauchy distribution
    and :math:`\operatorname{N}` is the normal/Gaussian distribution.

    The set of random variables,
    :math:`\mathcal{S} = \{\{\beta^{(k)}, \lambda^{(k)}, \tau^{(k)}, \tau^{(k)}, \pi^{(k)}\}_{k=1}^K, \{S_t\}^T_{t=1}\}`,
    are referred to as "stochastics" throughout the code.


    Parameters
    ==========
    y_data: pandas.Series or pandas.DataFrame
        Usage/response observations :math:`y_t`.
    X_data: list of pandas.DataFrame
        List of design matrices for each state, i.e. :math:`x_t^{(S_t)}`.  Each
        must span the entire length of observations (i.e. `y_data`).
    initial_params: NormalHMMInitialParams
        The initial parameters, which include
        :math:`\pi_0, m^{(k)}, \alpha^{(k)}, V^{(k)}`.
    single_obs_var: bool, optional
        Determines whether there are multiple observation variances or not.
        Only used when not given intial parameters.
    include_ppy: bool, optional
        If `True`, then include an unobserved observation Stochastic that can
        be used to produce posterior predicitve samples.  The Stochastic
        will have the name `y_pp`.

    Returns
    =======
    A ``pymc.Model`` object used for sampling.
    """

    N_states = len(X_data)
    N_obs = X_data[0].shape[0]

    alpha_trans = getattr(initial_params, 'alpha_trans', None)
    trans_mat_0 = getattr(initial_params, 'trans_mat', None)
    states_p_0 = getattr(initial_params, 'p0', None)
    states_0 = getattr(initial_params, 'states', None)
    betas_0 = getattr(initial_params, 'betas', None)
    V_invs_n_0 = getattr(initial_params, 'Vs_n', None)
    V_invs_S_0 = getattr(initial_params, 'Vs_S', None)
    V_invs_0 = getattr(initial_params, 'Vs', None)

    #
    # Some parameters can be set to arguably generic values
    # when no initial parameters are explicitly given.
    #
    if alpha_trans is None:
        alpha_trans = np.ones((N_states, N_states))

    if trans_mat_0 is None:
        trans_mat_0 = np.tile(1./N_states, (N_states, N_states-1))

    if betas_0 is None:
        betas_0 = [np.zeros(X_.shape[1]) for X_ in X_data]

    if V_invs_n_0 is None or V_invs_S_0 is None:
        V_invs_shape = (1 if single_obs_var else N_states,)
        if y_data is not None:
            V_invs_n_0 = np.tile(1, V_invs_shape)
            S_obs = np.clip(float(np.var(y_data)), 1e-4, np.inf)
            V_invs_S_0 = np.tile(S_obs, V_invs_shape)
            V_invs_0 = np.tile(S_obs, V_invs_shape)
        else:
            V_invs_n_0 = np.ones(1)
            V_invs_S_0 = np.tile(1e-3, V_invs_shape)
            V_invs_0 = np.tile(1e-3, V_invs_shape)

    # Transition probability stochastic:
    trans_mat = TransProbMatrix("trans_mat", alpha_trans,
                                value=trans_mat_0)

    # State sequence stochastics:
    states = HMMStateSeq("states", trans_mat, N_obs,
                         p0=states_p_0, value=states_0)

    # Observation precision distributions:
    V_inv_list = [pymc.Gamma('V-{}'.format(k),
                             n_0/2., n_0 * S_0/2.,
                             value=V_inv_0)
                  for k, (V_inv_0, n_0, S_0) in
                  enumerate(zip(V_invs_0, V_invs_n_0, V_invs_S_0))]

    V_invs = pymc.ArrayContainer(np.array(V_inv_list, dtype=np.object))

    beta_list = ()
    eta_list = ()
    lambda_list = ()
    beta_tau_list = ()
    mu_k_list = ()
    X_k_list = ()

    # This is a dynamic list of the time indices allocated to each state.
    # Very useful for breaking quantities mixtures into separate,
    # easy to handle problems.
    state_obs_idx = ()

    # Now, initialize all the intra-state terms:
    for k in range(N_states):

        k_idx = KIndex(k, states)

        state_obs_idx += (k_idx,)

        size_k = X_data[k].shape[1]
        size_k = size_k if size_k > 1 else None

        # Local shrinkage terms:
        lambda_k = pymc.Cauchy('lambdas-{}'.format(k),
                               0., 1., size=size_k)

        k_V = k if len(V_invs) == N_states else 0
        beta_tau_k = V_invs[k_V]

        beta_tau_k = beta_tau_k * lambda_k**2

        # We're initializing the scale of the global
        # shrinkage parameter's distribution with a decent,
        # asymptotically motivated value:
        if size_k is not None:
            # Global shrinkage term:
            eta_k = pymc.Cauchy('eta-{}'.format(k),
                                0., np.sqrt(np.log(size_k)))
            eta_list += (eta_k,)

            beta_tau_k = beta_tau_k * eta_k**2

        # Consider using just pymc.Cauchy; that way, there's
        # no support restriction that could make things difficult
        # for naive samplers, etc.

        beta_k = pymc.Normal('beta-{}'.format(k),
                             0., beta_tau_k,
                             value=betas_0[k],
                             size=size_k)

        #beta_k = pymc.TruncatedNormal('beta-{}'.format(k),
        #                              0, beta_tau_k,
        #                              0, np.inf,
        #                              value=betas_0[k],
        #                              size=size_k)

        X_k = NumpyTake(np.asarray(X_data[k]),
                        k_idx, axis=0)

        mu_k = pymc.LinearCombination("mu_{}".format(k),
                                      (X_k,), (beta_k,),
                                      trace=False)
        X_k_list += (X_k,)
        mu_k_list += (mu_k,)
        beta_list += (beta_k,)
        lambda_list += (lambda_k,)
        beta_tau_list += (beta_tau_k,)

    # These containers are handy, but not necessary.
    betas = pymc.TupleContainer(beta_list)
    etas = pymc.TupleContainer(eta_list)
    lambdas = pymc.TupleContainer(lambda_list)
    beta_taus = pymc.TupleContainer(beta_tau_list)

    mu = NumpyHstack(mu_k_list)
    #mu = HMMLinearCombination('mu', X_data, betas, states)

    if np.alen(V_invs) == 1:
        V_inv = V_invs[0]
    else:
        V_inv = NumpyChoose(states, V_inv_list)

    y_observed = False
    if y_data is not None:
        y_observed = True
        y_data = np.ma.masked_invalid(y_data).astype(np.object)
        y_data.set_fill_value(None)

    y_rv = pymc.Normal('y', mu, V_inv,
                       value=y_data,
                       observed=y_observed)

    #y_rv = pymc.TruncatedNormal('y', mu, V_inv,
    #                            0., np.inf,
    #                            value=y_data,
    #                            observed=y_observed)

    if y_observed and include_ppy:
        y_pp_rv = pymc.Normal('y_pp', mu, V_inv, trace=True)

    return pymc.Model(locals())


def make_poisson_hmm(y_data, X_data, initial_params):
    r""" Construct a PyMC2 scalar poisson-emmisions HMM model.

    TODO: Update to match normal model design.

    The model takes the following form:

    .. math::

        y_t &\sim \operatorname{Poisson}(\exp(x_t^{(S_t)\top} \beta^{(S_t)})) \\
        \beta^{(S_t)}_i &\sim \operatorname{N}(m^{(S_t)}, C^{(S_t)}),
        \quad i \in \{1,\dots,M\} \\
        S_t \mid S_{t-1} &\sim \operatorname{Categorical}(\pi^{(S_{t-1})}) \\
        \pi^{(S_t-1)} &\sim \operatorname{Dirichlet}(\alpha^{(S_{t-1})})

    where :math:`C^{(S_t)} = \lambda_i^{(S_t) 2} \tau^{(S_t) 2}` and

    .. math::

        \lambda^{(S_t)}_i &\sim \operatorname{Cauchy}^{+}(0, 1) \\
        \tau^{(S_t)} &\sim \operatorname{Cauchy}^{+}(0, 1)

    for observations :math:`y_t` in :math:`t \in \{0, \dots, T\}`,
    features :math:`x_t^{(S_t)} \in \mathbb{R}^M`,
    regression parameters :math:`\beta^{(S_t)}`, state sequences
    :math:`\{S_t\}^T_{t=1}` and
    state transition probabilities :math:`\pi \in [0, 1]^{K}`.
    :math:`\operatorname{Cauchy}^{+}` is the standard half-Cauchy distribution
    and :math:`\operatorname{N}` is the normal/Gaussian distribution.

    The set of random variables,
    :math:`\mathcal{S} = \{\{\beta^{(k)}, \lambda^{(k)}, \tau^{(k)}, \tau^{(k)}, \pi^{(k)}\}_{k=1}^K, \{S_t\}^T_{t=1}\}`,
    are referred to as "stochastics" throughout the code.


    Parameters
    ==========
    y_data: pandas.DataFrame
        Usage/response observations :math:`y_t`.
    X_data: list of pandas.DataFrame
        List of design matrices for each state, i.e. :math:`x_t^{(S_t)}`.  Each
        must span the entire length of observations (i.e. `y_data`).
    initial_params: NormalHMMInitialParams
        The initial parameters, which include
        :math:`\pi_0, m^{(k)}, \alpha^{(k)}, V^{(k)}`.
        Ignores `V` parameters.
        FIXME: using the "Normal" initial params objects is only temporary.

    Returns
    =======
    A ``pymc.Model`` object used for sampling.
    """

    N_states = len(X_data)
    N_obs = X_data[0].shape[0]

    alpha_trans = initial_params.alpha_trans

    trans_mat = TransProbMatrix("trans_mat", alpha_trans,
                                value=initial_params.trans_mat)

    states = HMMStateSeq("states", trans_mat, N_obs, p0=initial_params.p0,
                         value=initial_params.states)

    betas = []
    etas = []
    lambdas = []
    for s in range(N_states):

        initial_beta = None
        if initial_params.betas is not None:
            initial_beta = initial_params.betas[s]

        size_s = X_data[s].shape[1]
        size_s = size_s if size_s > 1 else None

        lambda_s = pymc.HalfCauchy('lambda-{}'.format(s),
                                   0., 1., size=size_s)

        eta_s = pymc.HalfCauchy('tau-{}'.format(s),
                                0., 1.)

        beta_s = pymc.Normal('beta-{}'.format(s),
                             0., (lambda_s * eta_s)**(-2),
                             value=initial_beta,
                             size=size_s)

        betas += [beta_s]
        etas += [eta_s]
        lambdas += [lambda_s]

    mu_reg = HMMLinearCombination('mu', X_data, betas, states, trace=False)

    @pymc.deterministic(trace=True, plot=False)
    def mu(mu_reg_=mu_reg):
        return np.exp(mu_reg_)

    if y_data is not None:
        y_data = np.ma.masked_invalid(y_data).astype(np.object)
        y_data.set_fill_value(None)

    y_rv = pymc.Poisson('y', mu, value=y_data,
                        observed=True if y_data is not None else False)

    del initial_params, s, beta_s, size_s, lambda_s, eta_s

    return pymc.Model(locals())


def make_normal_baseline_hmm(y_data, X_data, baseline_end, initial_params):
    """ Construct a PyMC2 scalar normal-emmisions HMM with a
    stochastic reporting period start time parameter and baseline, reporting
    parameters for all other stochastics/estimated terms in the model.
    The reporting period start time parameter is given a discrete uniform
    distribution starting from the first observation after the baseline to the
    end of the series.

    Parameters
    ==========
    y_data: pandas.DataFrame
        Usage/response observations.
    X_data: list of pandas.DataFrame
        List of design matrices for each state.  Each must
        span the entire length of observations (i.e. `y_data`).
    baseline_end: pandas.tslib.Timestamp
        End of baseline period (inclusive), beginning of reporting period.
    initial_params: NormalHMMInitialParams
        An object containing the following fields/members:
    Returns
    =======
    A pymc.Model object used for sampling.
    """

    N_states = len(X_data)
    N_obs = X_data[0].shape[0]

    alpha_trans = initial_params.alpha_trans

    # TODO: If we wanted a distribution over the time
    # when a renovation becomes effective...
    baseline_idx = X_data[0].index.get_loc(baseline_end)
    reporting_start = pymc.DiscreteUniform("reporting_start",
                                           baseline_idx + 1,
                                           N_obs,
                                           value=baseline_idx + 1)

    trans_mat_baseline = TransProbMatrix("trans_mat_baseline", alpha_trans,
                                         value=initial_params.trans_mat)
    trans_mat_reporting = TransProbMatrix("trans_mat_reporting", alpha_trans,
                                          value=initial_params.trans_mat)

    @pymc.deterministic(trace=True, plot=False)
    def N_baseline(rs_=reporting_start):
        return rs_ - 1
    states_baseline_0 = initial_params.states[slice(0, baseline_idx)]
    states_baseline = HMMStateSeq("states_baseline", trans_mat_baseline,
                                  N_baseline, p0=initial_params.p0,
                                  value=states_baseline_0)

    @pymc.deterministic(trace=True, plot=False)
    def N_reporting(rs_=reporting_start):
        return N_obs - rs_
    states_reporting_0 = initial_params.states[slice(baseline_idx, N_obs)]
    # TODO, FIXME: p0 should depend on states_baseline and trans_mat_baseline,
    # no?
    states_reporting = HMMStateSeq("states_reporting", trans_mat_reporting,
                                   N_reporting, p0=initial_params.p0,
                                   value=states_reporting_0)

    @pymc.deterministic(trace=True, plot=False)
    def states(sb_=states_baseline, sr_=states_reporting):
        return np.concatenate([sb_, sr_])

    Ws = initial_params.Ws
    betas = [[], []]
    for s in range(N_states):
        size_s = len(initial_params.betas[s])
        baseline_beta_s = pymc.Cauchy('base-beta-{}'.format(s),
                                      initial_params.betas[s], Ws[s],
                                      value=initial_params.betas[s],
                                      size=size_s if size_s > 1 else None)
        betas[0] += [baseline_beta_s]

        reporting_beta_s = pymc.Cauchy('rep-beta-{}'.format(s),
                                       initial_params.betas[s], Ws[s],
                                       value=initial_params.betas[s],
                                       size=size_s if size_s > 1 else None)
        betas[1] += [reporting_beta_s]

    del s, baseline_beta_s, reporting_beta_s, size_s

    Vs = initial_params.Vs

    mu = HMMLinearCombination('mu', X_data, betas, states)

    @pymc.deterministic(trace=False, plot=False)
    def V(states_=states, V_=Vs):
        return V_[states_]

    if y_data is not None:
        y_data = np.ma.masked_invalid(y_data).astype(np.object)
        y_data.set_fill_value(None)

    y_rv = pymc.Normal('y', mu, 1./V, value=y_data,
                       observed=True if y_data is not None else False)

    del initial_params

    return pymc.Model(locals())
