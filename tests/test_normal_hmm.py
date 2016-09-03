from __future__ import division

import sys
from functools import partial

import pandas as pd
import numpy as np
import pymc

import pytest
from numpy.testing import assert_array_less, assert_array_equal
from amimodels.testing import assert_hpd

from amimodels.normal_hmm import (NormalHMMProcess, make_normal_hmm,
                                  compute_trans_freqs,
                                  gmm_norm_hmm_init_params,
                                  bic_norm_hmm_init_params,
                                  NormalHMMInitialParams,
                                  calc_alpha_prior, trace_sampler,
                                  get_stochs_excluding,
                                  generate_temperatures)
from amimodels.step_methods import (
    TransProbMatStep,
    HMMStatesStep,
    NormalNormalStep)

if hasattr(sys, '_called_from_test'):
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
    )


def plot_mean_predictions(mcmc_step, ram_db, y_obs, y_oos_df):
    mu_pred_df = pd.DataFrame(ram_db.trace('mu').gettrace().T,
                              index=y_oos_df.index)
    mu_pred_mean_df = mu_pred_df.mean(axis=1)
    mu_pred_mean_df.columns = [r'$E[\mu_t]$ pred']

    states_pred_df = pd.DataFrame(ram_db.trace('states').gettrace().T,
                                  index=y_oos_df.index) + 1
    states_pred_mean = states_pred_df.mean(axis=1)
    states_pred_mean_df = pd.DataFrame(states_pred_mean,
                                       index=y_oos_df.index,
                                       columns=[r'$E[S_t]$ pred'])

    from amimodels.hmm_utils import plot_hmm
    axes = None
    #_ = [ax_.clear() for ax_ in axes]
    axes = plot_hmm(mcmc_step, obs_index=y_obs.index, axes=axes)

    y_oos_df.columns = [r'$y_t$ o.o.s.']
    y_oos_df.plot(ax=axes[0], drawstyle='steps-mid')

    mu_pred_df.plot(ax=axes[0], drawstyle='steps-mid',
                    alpha=8/mu_pred_df.shape[1],
                    rasterized=True, color='lightgreen',
                    legend=False)

    mu_pred_mean_df.plot(ax=axes[0], drawstyle='steps-mid',
                         color='green')

    states_pred_df.plot(ax=axes[1], drawstyle='steps-mid',
                        alpha=8/mu_pred_df.shape[1],
                        rasterized=True, color='lightgreen',
                        legend=False)

    states_pred_mean_df.plot(ax=axes[1], drawstyle='steps-mid',
                             color='green')


@pytest.fixture(params=[np.asarray([[0.9], [0.2]]),
                        np.asarray([[0.9], [0.2]]),
                        np.asarray([[0.1], [0.8]]),
                        np.asarray([[0.5], [0.5]]),
                        np.asarray([[0.9], [0.9]]),
                        np.asarray([[0.1], [0.3]])])
def model_true(request):
    model_true = NormalHMMProcess(request.param,
                                  500,
                                  np.asarray([1., 0.]),
                                  np.asarray([[0.2], [1.]]),
                                  np.asarray([0.1/8.**2, 0.5/3.**2]),
                                  seed=249298)
    return model_true


@pytest.fixture(params=[gmm_norm_hmm_init_params,
                        bic_norm_hmm_init_params])
def init_func(request):
    return request.param


@pytest.mark.parametrize("test_input,expected", [
    (pd.DataFrame(np.asarray([1, 0, 0, 0, 1])), np.array([2, 1, 1, 0])),
    (pd.DataFrame(np.asarray([1, np.nan, 0, 0, 1])), np.array([1, 1, 0, 0]))])
def test_compute_trans_freqs(test_input, expected):
    assert_array_equal(expected,
                       compute_trans_freqs(test_input, 2, True).flatten())


def test_observed_trans_probs(model_true):
    """ Let's check some basic results regarding the simulated transition
    probabilities.  Specifically, how closely do the empirical values of
    :math: `P(S_t = i \mid S_{t-1}=j)` match the true ones?
    """

    np.random.seed(2352523)
    states_true, y, X_matrices = model_true.simulate()

    P_obs = compute_trans_freqs(states_true, model_true.P.shape[0])
    trans_mat_obs = P_obs[:, 0]
    trans_mat_true = model_true.trans_mat.flatten()

    assert_array_less(np.abs(trans_mat_obs - trans_mat_true),
                      1./np.sqrt(states_true.size//2) +
                      np.sqrt(trans_mat_true * (1-trans_mat_true)))


@pytest.mark.skip(reason="In progress...")
@slow
def test_estimated_trans_probs(model_true,
                               init_func,
                               mcmc_iters=2000):
    """ Check that the estimated mean transition probabilities
    are within range of the true ones.
    """

    np.random.seed(2352523)
    states_true, y, X_matrices = model_true.simulate()

    init_params = init_func(y, X_matrices)

    norm_hmm = make_normal_hmm(y, X_matrices, init_params)

    norm_hmm.states.save_trace = False

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)
    mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)
    for b_ in norm_hmm.betas:
        mcmc_step.use_step_method(NormalNormalStep, b_)

    mcmc_step.sample(mcmc_iters)

    #if isinstance(trans_mat_step, pymc.StepMethods.Metropolis):
    #    assert trans_mat_step.ratio > 0.0,\
    #        "trans_mat_step acceptance rate <= 0.0"

    #if isinstance(states_step, pymc.StepMethods.Metropolis):
    #    assert states_step.ratio > 0.0,\
    #        "states_step acceptance rate <= 0.0"

    trans_mat_true = model_true.trans_mat

    assert_hpd(norm_hmm.trans_mat, trans_mat_true, alpha=0.01)


def test_missing_init(init_func):
    """Test that initial states are sampled for masked/missing
    observations.
    """

    np.random.seed(2352523)
    model_true = NormalHMMProcess(np.asarray([[0.4], [0.1]]),
                                  20,
                                  np.asarray([1., 0.]),
                                  np.asarray([[0.2], [1., 0.04]]),
                                  np.asarray([0.1/8.**2, 0.5/3.**2]),
                                  formulas=["1 ~ 1", "1 ~ 1 + temp"],
                                  seed=249298)

    states_true, y, X_matrices = model_true.simulate()

    baseline_end = y.index[int(y.size * 0.75)]
    y_mask = y.eval('index > @baseline_end or index == index.min()')
    y_obs = y.mask(y_mask)

    init_params = init_func(y_obs, X_matrices)

    assert not any(init_params.states.isnull())


def test_degenerate(mcmc_iters=200):
    """Test that the initial value are reasonable for degenerate
    observations.
    """
    np.random.seed(2352523)
    N_obs = 20
    y_obs = pd.DataFrame(np.ones(N_obs))
    X_matrices = [pd.DataFrame(np.ones((N_obs, 1))),
                  pd.DataFrame(np.ones((N_obs, 2)))]

    norm_hmm = make_normal_hmm(y_obs, X_matrices)

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    #from itertools import chain
    #for e_ in chain(norm_hmm.etas, norm_hmm.lambdas):
    #    mcmc_step.use_step_method(pymc.StepMethods.Metropolis,
    #                              e_, proposal_distribution='Prior')

    mcmc_step.sample(mcmc_iters)

    assert_hpd(norm_hmm.mu, np.ones(N_obs))
    # We don't necessarily know the order of the states (i.e.
    # does state 1 have smallest mean, etc.), so it's hard
    # to make the comparison without imposing an order, and
    # that's not always [distinctly] possible (e.g. more than one
    # non-ordered parameter in a state).

    #assert_hpd(norm_hmm.states, np.zeros(N_obs))
    #assert_hpd(norm_hmm.betas[0], np.ones(norm_hmm.betas[0].shape))
    #assert_hpd(norm_hmm.betas[1], np.zeros(norm_hmm.betas[1].shape))


def test_no_est(model_true,
                mcmc_iters=200):
    """ Initialize a model with some good/correct values,
    do *not* estimate anything (i.e. don't use observations) and
    make sure that it can sample the model reasonably well.
    """

    np.random.seed(2352523)

    states_true, y, X_matrices = model_true.simulate()

    y_obs = None
    N_states = len(X_matrices)
    N_obs = X_matrices[0].shape[0]
    trans_mat_full = np.column_stack(
        (model_true.trans_mat, 1. - model_true.trans_mat.sum(axis=1)))

    alpha_trans = calc_alpha_prior(states_true, N_states,
                                   trans_mat_full)

    init_params = NormalHMMInitialParams(alpha_trans,
                                         model_true.trans_mat,
                                         states_true,
                                         model_true.betas,
                                         None,
                                         model_true.Vs,
                                         None)

    norm_hmm = make_normal_hmm(y_obs, X_matrices, init_params)

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.sample(mcmc_iters)

    # TODO: Check all estimated quantities?
    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)
    assert_hpd(norm_hmm.states, states_true)
    assert_hpd(norm_hmm.betas[0], model_true.betas[0])
    assert_hpd(norm_hmm.betas[1], model_true.betas[1])


@slow
def test_missing_inference(model_true,
                           init_func,
                           mcmc_iters=200):
    """ Test that the estimation and initialization can handle missing data.
    """

    states_true, y, X_matrices = model_true.simulate()

    baseline_end = y.index[int(y.size * 0.75)]
    N_obs_half = y.size // 2
    y_mask = y.eval('index > @baseline_end or index == index.min()\
                    or index == index[@N_obs_half]')
    y_obs = y.mask(y_mask)

    init_params = init_func(y_obs, X_matrices)

    norm_hmm = make_normal_hmm(y, X_matrices, init_params)

    assert any(y_obs.isnull())

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)
    mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)
    for b_ in norm_hmm.betas:
        mcmc_step.use_step_method(NormalNormalStep, b_)

    mcmc_step.sample(mcmc_iters)

    # TODO: Check all estimated quantities?
    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)


def test_sinusoid():
    """ Generate a two-state normal emissions HMM with
    the first state having a low constant value and the second
    a regression involving a sinusoidal exogenous variable.
    We fit the same model but include an extraneous variable to the
    regression.
    """
    temperature_generator = partial(generate_temperatures,
                                    period=23.,
                                    offset=np.pi,
                                    base_temp=50.,
                                    flux_amt=30.)
    model_true = NormalHMMProcess(np.asarray([[0.8], [0.08]]),
                                  300,
                                  np.asarray([1., 0.]),
                                  np.asarray([[0.2], [0.2, 0.01, 0.]]),
                                  np.asarray([0.1 / 10.**3, 0.5 / 10.**2]),
                                  exogenous_sim_func=temperature_generator,
                                  formulas=["1 ~ 1",
                                            "1 ~ 1 + temp + I(temp**2)"],
                                  seed=249298)

    states_true, y, X_matrices = model_true.simulate()

    means_true = np.fromiter((np.dot(X_matrices[s].iloc[t],
                                     model_true.betas[s])
                              for t, s in enumerate(states_true)),
                             dtype=np.float)

    norm_hmm = make_normal_hmm(y, X_matrices, None,
                               single_obs_var=True)

    mcmc_step = pymc.MCMC(norm_hmm.variables)
    mcmc_step.assign_step_methods()
    #mcmc_step.step_method_dict

    mcmc_iters = 200
    mcmc_step.sample(mcmc_iters, burn=mcmc_iters//2)

    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)
    assert_hpd(norm_hmm.states, states_true)

    assert_hpd(norm_hmm.mu, means_true, rtol=0.15)

    assert_hpd(norm_hmm.betas[0], model_true.betas[0])

    assert_hpd(norm_hmm.betas[1], model_true.betas[1],
               rtol=np.array([0.25, 0.1, 0.1]))



@pytest.mark.skip(reason="In progress...")
def test_prediction(
                    mcmc_iters=200
                    ):
    """ Test out-of-sample/posterior predictive sampling.
    """

    np.random.seed(2352523)

    trans_mat = np.array([[0.9], [0.2]])

    model_true = NormalHMMProcess(trans_mat,
                                  500,
                                  np.asarray([1., 0.]),
                                  np.asarray([[0.2], [1.]]),
                                  np.asarray([0.1/8.**2, 0.5/3.**2]),
                                  seed=249298)

    states_true, y, X_matrices = model_true.simulate()
    y_obs = y

    # DEBUG: Trying to suss out a random numpy indexing warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        norm_hmm = make_normal_hmm(y_obs, X_matrices,
                                   single_obs_var=False)

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    from itertools import chain
    for e_ in chain(norm_hmm.etas, norm_hmm.lambdas):
        mcmc_step.use_step_method(pymc.StepMethods.Metropolis,
                                  e_, proposal_distribution='Prior')

    mcmc_step.sample(mcmc_iters)

    # DEBUG: Remove.
    get_ipython().magic('matplotlib qt')
    from amimodels.hmm_utils import plot_hmm
    axes = None
    if axes is not None:
        _ = [ax_.clear() for ax_ in axes]
    axes = plot_hmm(mcmc_step, obs_index=y_obs.index, axes=axes,
                    plot_samples=False)

    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)
    assert_hpd(norm_hmm.states, states_true)
    #assert_hpd(norm_hmm.betas[0], model_true.betas[0])
    #assert_hpd(norm_hmm.betas[1], model_true.betas[1])

    #
    # Save the trace values for the variable we want
    # to predict.
    #
    non_time_parents = get_stochs_excluding(norm_hmm.mu,
                                            set(('states', 'N_obs')))
    traces = {}
    for stoch in non_time_parents:
        traces[stoch.__name__] = mcmc_step.trace(stoch).gettrace()

    #
    # Generate new design matrices for predictions over
    # new time intervals.
    #
    model_true.start_datetime = y_obs.index[-1]
    model_true.N_obs = model_true.N_obs // 2
    states_oos_df, y_oos_df, X_mat_pred = model_true.simulate()

    # Initialize a new model using the new design matrices,
    # but **not** the new observations.
    norm_hmm_pred = make_normal_hmm(None, X_mat_pred)

    ram_db = trace_sampler(norm_hmm_pred, 'mu', traces)

    # DEBUG: Remove.
    #plot_mean_predictions(mcmc_step, ram_db, y_obs, y_oos_df)

    mu_pred_df = pd.DataFrame(ram_db.trace('mu').gettrace().T,
                              index=y_oos_df.index)
    # TODO: Not easy to compare these unless we have very strong
    # predictors.  Even then, the states will vary freely.
    #sqrd_err = np.abs(mu_pred_df.values - y_oos_df.values).mean()


@pytest.mark.skip(reason="In progress...")
def test_example_1():
    """ TODO
    """
    import pickle
    import patsy

    model_data, formulas = pickle.load(open('tests/data/test_example_1.pkl', 'rb'))

    X_matrices = []
    for formula in formulas:
        y, X = patsy.dmatrices(formula, model_data,
                               return_type='dataframe')
        X_matrices += [X]

    init_params = None  # gmm_norm_hmm_init_params(y, X_matrices)

    norm_hmm = make_normal_hmm(y, X_matrices, init_params,
                                include_ppy=True)

    # Some very basic, generic initializations for the state parameters.
    norm_hmm.betas[0].value = 0
    b1_0 = norm_hmm.betas[1].value.copy()
    b1_0[0] = float(y.mean())
    norm_hmm.betas[1].value = b1_0

    mcmc_step = pymc.MCMC(norm_hmm.variables)
    mcmc_step.assign_step_methods()

    # TODO: Check results.

