from __future__ import division

import pandas as pd
import numpy as np
import pymc

import pytest
from numpy.testing import assert_array_less, assert_array_equal
from amimodels.testing import assert_hpd

from amimodels.normal_hmm import (NormalHMMProcess, make_normal_hmm,
                                  compute_trans_freqs, gmm_norm_hmm_init_params,
                                  bic_norm_hmm_init_params, NormalHMMInitialParams)
from amimodels.step_methods import TransProbMatStep, HMMStatesStep

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


@pytest.fixture(params=[np.asarray([[0.9], [0.2]]),
                        np.asarray([[0.9], [0.2]]),
                        np.asarray([[0.1], [0.8]]),
                        np.asarray([[0.5], [0.5]]),
                        np.asarray([[0.9], [0.9]]),
                        np.asarray([[0.1], [0.3]])])
def process_2_state_trans(request):
    model_true = NormalHMMProcess(request.param,
                                  500,
                                  np.asarray([1., 0.]),
                                  np.asarray([[0.2], [1.]]),
                                  np.asarray([0.1/8.**2, 0.5/3.**2]),
                                  seed=249298)
    return model_true


@pytest.fixture(params=[gmm_norm_hmm_init_params,
                        bic_norm_hmm_init_params])
def initialize_2_state_trans(request):
    return request.param


@pytest.mark.parametrize("test_input,expected", [
    (pd.DataFrame(np.asarray([1, 0, 0, 0, 1])), np.array([2, 1, 1, 0])),
    (pd.DataFrame(np.asarray([1, np.nan, 0, 0, 1])), np.array([1, 1, 0, 0]))])
def test_compute_trans_freqs(test_input, expected):
    assert_array_equal(expected,
                       compute_trans_freqs(test_input, 2, True).flatten())


def test_observed_trans_probs(process_2_state_trans):
    """ Let's check some basic results regarding the simulated transition
    probabilities.  Specifically, how closely do the empirical values of
    :math: `P(S_t = i \mid S_{t-1}=j)` match the true ones?
    """

    states_true, y, X_matrices = process_2_state_trans.simulate()

    P_obs = compute_trans_freqs(states_true, process_2_state_trans.P.shape[0])
    trans_mat_obs = P_obs[:, 0]
    trans_mat_true = process_2_state_trans.trans_mat.flatten()

    assert_array_less(np.abs(trans_mat_obs - trans_mat_true),
                      1./np.sqrt(states_true.size//2) +
                      np.sqrt(trans_mat_true * (1-trans_mat_true)))


@pytest.mark.skip(reason="In progress...")
@slow
def test_estimated_trans_probs(process_2_state_trans,
                               initialize_2_state_trans,
                               mcmc_iters=2000,
                               progress_bar=False):
    """ Check that the estimated mean transition probabilities
    are within range of the true ones.
    """

    states_true, y, X_matrices = process_2_state_trans.simulate()

    init_params = initialize_2_state_trans(y, X_matrices)

    norm_hmm = make_normal_hmm(y, X_matrices, init_params)

    norm_hmm.states.save_trace = False

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)

    mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)

    mcmc_step.sample(mcmc_iters,
                     burn=mcmc_iters//2,
                     progress_bar=progress_bar,
                     verbose=0)

    #if isinstance(trans_mat_step, pymc.StepMethods.Metropolis):
    #    assert trans_mat_step.ratio > 0.0,\
    #        "trans_mat_step acceptance rate <= 0.0"

    #if isinstance(states_step, pymc.StepMethods.Metropolis):
    #    assert states_step.ratio > 0.0,\
    #        "states_step acceptance rate <= 0.0"

    trans_mat_true = process_2_state_trans.trans_mat

    assert_hpd(norm_hmm.trans_mat, trans_mat_true, alpha=0.01)


def test_missing_init(initialize_2_state_trans):
    """Test that initial states are sampled for masked/missing
    observations.
    """

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

    init_params = initialize_2_state_trans(y_obs, X_matrices)

    assert not any(init_params.states.isnull())


@pytest.mark.skip(reason="In progress...")
@slow
def test_prediction(process_2_state_trans,
                    initialize_2_state_trans,
                    mcmc_iters=2000,
                    progress_bar=False):

    # TODO: testing; remove.
    from collections import namedtuple
    Request = namedtuple('Request', ['param'])
    r = Request(param=np.asarray([[0.9], [0.2]]))
    model_true = process_2_state_trans(r)

    states_true, y, X_matrices = model_true.simulate()

    y_obs = None
    N_states = len(X_matrices)
    N_obs = X_matrices[0].shape[0]

    init_params = NormalHMMInitialParams(np.ones((N_states, N_states)),
                                         None,
                                         np.zeros(N_obs),
                                         None,
                                         None,
                                         model_true.Vs,
                                         None)

    norm_hmm = make_normal_hmm(y_obs, X_matrices, init_params)

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.assign_step_methods(verbose=100)

    mcmc_iters = 2000
    progress_bar = True
    mcmc_step.sample(mcmc_iters,
                     burn=mcmc_iters//2,
                     progress_bar=progress_bar)

    # TODO: Check all estimated quantities?
    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)


@slow
def test_missing_inference(process_2_state_trans,
                           initialize_2_state_trans,
                           mcmc_iters=2000,
                           progress_bar=False):
    """ Test that the estimation and initialization can handle missing data.
    """

    states_true, y, X_matrices = model_true.simulate()

    baseline_end = y.index[int(y.size * 0.75)]
    N_obs_half = y.size // 2
    y_mask = y.eval('index > @baseline_end or index == index.min()\
                    or index == index[@N_obs_half]')
    y_obs = y.mask(y_mask)

    init_params = initialize_2_state_trans(y_obs, X_matrices)

    norm_hmm = make_normal_hmm(y, X_matrices, init_params)

    assert any(y_obs.isnull())

    mcmc_step = pymc.MCMC(norm_hmm.variables)

    mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)

    mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)

    mcmc_iters = 2000
    progress_bar = True
    mcmc_step.sample(mcmc_iters,
                     burn=mcmc_iters//2,
                     progress_bar=progress_bar)

    # TODO: Check all estimated quantities?
    assert_hpd(norm_hmm.trans_mat, model_true.trans_mat)
