from __future__ import division

import sys

import numpy as np
import pymc

from numpy.testing import (assert_allclose,
                           assert_array_less,
                           assert_array_equal,
                           assert_equal)
import pytest

from amimodels.stochastics import HMMStateSeq, TransProbMatrix
from amimodels.step_methods import (HMMStatesStep, TransProbMatStep,
                                    NormalNormalStep, PriorObsSampler)
from amimodels.testing import *

import warnings

warnings.filterwarnings('always')

if hasattr(sys, '_called_from_test'):
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
    )


def test_states_single_step():
    """Test custom sampling of mixture states (in isolation).
    """

    np.random.seed(2352523)

    model_true = simple_state_seq_model()

    trans_mat_true = model_true['trans_mat_obs']
    mu_true = model_true['mu'].value
    states_true = model_true['states_rv'].value
    y_obs = model_true['y_rv'].value

    model_test = simple_state_seq_model(y_obs=y_obs)

    mcmc_step = pymc.MCMC(model_test)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, model_test['states_rv'])

    (states_step,) = mcmc_step.step_method_dict[model_test['states_rv']]
    assert isinstance(states_step, HMMStatesStep)

    #
    # First, let's check that the logp function is working.
    #
    y_logps_0_true = np.array([pymc.normal_like(y_obs[t],
                                                model_test['mu_vals'][0], 100)
                               for t in xrange(model_test['N_obs'])])
    y_logps_1_true = np.array([pymc.normal_like(y_obs[t],
                                                model_test['mu_vals'][1], 100)
                               for t in xrange(model_test['N_obs'])])

    states_step.compute_y_logp()
    y_logps_est = states_step.y_logp_vals

    assert_allclose(y_logps_est[0], y_logps_0_true)
    assert_allclose(y_logps_est[1], y_logps_1_true)

    mcmc_step.sample(2000)

    assert_hpd(mcmc_step.states_rv, states_true, alpha=0.01)


def test_trans_prob_mat_step():
    """Exclusively test transition probability matrix step method.
    The states are held fixed/observations and nothing else is
    estimated.
    """

    np.random.seed(2352523)

    model_true = simple_state_trans_model()

    trans_mat_true = model_true['trans_mat_rv'].value
    mu_true = model_true['mu'].value
    states_true = model_true['states_rv'].value
    y_obs = model_true['y_rv'].value

    model_test = simple_state_trans_model(y_obs=y_obs)

    mcmc_step = pymc.MCMC(model_test)
    mcmc_step.draw_from_prior()

    assert not np.allclose(mcmc_step.trans_mat_rv.value, trans_mat_true)

    mcmc_step.use_step_method(TransProbMatStep, mcmc_step.trans_mat_rv)
    (trans_mat_step, ) = mcmc_step.step_method_dict[mcmc_step.trans_mat_rv]
    assert isinstance(trans_mat_step, TransProbMatStep)

    mcmc_step.sample(2000)

    assert_hpd(mcmc_step.trans_mat_rv, trans_mat_true, alpha=0.01)


def test_states_trans_steps():
    """Test sampling of mixture states and transition matrices
    exclusively and simultaneously.  I.e. no regression terms/means
    or variances to estimate, only the state sequence and transition
    probability matrix.
    """

    np.random.seed(2352523)

    model_true = simple_state_trans_model()

    trans_mat_true = model_true['trans_mat_rv'].value
    mu_true = model_true['mu'].value
    states_true = model_true['states_rv'].value
    y_obs = model_true['y_rv'].value

    model_test = simple_state_trans_model(y_obs=y_obs)

    mcmc_step = pymc.MCMC(model_test)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, mcmc_step.states_rv)
    (states_step, ) = mcmc_step.step_method_dict[mcmc_step.states_rv]
    assert isinstance(states_step, HMMStatesStep)

    mcmc_step.use_step_method(TransProbMatStep, mcmc_step.trans_mat_rv)
    (trans_mat_step, ) = mcmc_step.step_method_dict[mcmc_step.trans_mat_rv]
    assert isinstance(trans_mat_step, TransProbMatStep)

    #
    # First, let's check that the logp function is working.
    #
    y_logps_0_true = np.array([pymc.normal_like(y_obs[t],
                                                model_test['mu_vals'][0], 100)
                               for t in xrange(model_test['N_obs'])])
    y_logps_1_true = np.array([pymc.normal_like(y_obs[t],
                                                model_test['mu_vals'][1], 100)
                               for t in xrange(model_test['N_obs'])])

    states_step.compute_y_logp()
    y_logps_est = states_step.y_logp_vals

    assert_allclose(y_logps_est[0], y_logps_0_true)
    assert_allclose(y_logps_est[1], y_logps_1_true)

    mcmc_step.sample(2000)

    assert_hpd(mcmc_step.states_rv, states_true, alpha=0.1)
    assert_hpd(mcmc_step.trans_mat_rv, trans_mat_true, alpha=0.1)


@pytest.mark.skip(reason="In progress...")
def test_states_multiple_step():
    """ Test that the state sequence step method works when combining separate
    state sequence stochastics (e.g. for change-point/structural change
    models).
    """

    np.random.seed(2352523)
    alpha_trans = np.asarray([[1, 1], [1, 1]])
    trans_mat = TransProbMatrix('trans_mat', alpha_trans)
    trans_mat_true = trans_mat.random()
    N_obs = 100
    states_1_rv = HMMStateSeq("states_1", trans_mat_true, N_obs // 2)
    states_2_rv = HMMStateSeq("states_2", trans_mat_true, N_obs - N_obs // 2)

    states_1_true = states_1_rv.value
    states_2_true = states_2_rv.value

    @pymc.deterministic(trace=True, plot=False)
    def states(states_1_=states_1_rv, states_2_=states_2_rv):
        return np.concatenate([states_1_, states_1_])

    @pymc.deterministic(trace=True, plot=False)
    def mu(states_=states):
        return np.asarray([-1, 1])[states_]

    states_true = states.value
    y_obs_rv = pymc.Normal('y_obs', mu, 100)
    y_obs = y_obs_rv.random()

    y_rv = pymc.Normal('y', mu, 100, value=y_obs, observed=True)

    norm_hmm = {'states_1': states_1_rv, 'states_2': states_2_rv, 'y': y_rv}

    mcmc_step = pymc.MCMC(norm_hmm)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, [states_1_rv, states_2_rv])

    (states_1_step, ) = mcmc_step.step_method_dict[mcmc_step.states_1]
    (states_2_step, ) = mcmc_step.step_method_dict[mcmc_step.states_2]

    assert isinstance(states_1_step, HMMStatesStep) and\
        isinstance(states_2_step, HMMStatesStep)

    assert states_1_step is states_2_step

    mcmc_step.sample(2000)

    assert_hpd(mcmc_step.states_1, states_1_true, alpha=0.01)
    assert_hpd(mcmc_step.states_2, states_2_true, alpha=0.01)


def test_states_missing():
    """ Test that the state sequence step function properly handles missing
    observations.
    """
    np.random.seed(2352523)

    model_true = simple_state_seq_model()

    trans_mat_true = model_true['trans_mat_obs']
    mu_true = model_true['mu'].value
    states_true = model_true['states_rv'].value
    y_true = model_true['y_rv'].value

    N_obs_half = model_true['N_obs'] // 2
    y_mask = np.arange(model_true['N_obs']) > N_obs_half
    y_obs = np.ma.masked_array(y_true, mask=y_mask)

    model_test = simple_state_seq_model(y_obs=y_obs)

    mcmc_step = pymc.MCMC(model_test)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, model_test['states_rv'])

    (states_step,) = mcmc_step.step_method_dict[model_test['states_rv']]
    assert isinstance(states_step, HMMStatesStep)

    #
    # First, let's check that the logp function is working.
    # In this case, that also means missing values were forward-filled.
    #
    ind = np.arange(np.alen(y_obs))
    ind[y_obs.mask] = 0

    y_logps_0_true = np.array([pymc.normal_like(y, model_test['mu_vals'][0],
                                                100) for t, y in
                               enumerate(y_obs)])
    y_logps_0_true[y_obs.mask] = 0.
    y_logps_1_true = np.array([pymc.normal_like(y, model_test['mu_vals'][1],
                                                100) for t, y in
                               enumerate(y_obs)])
    y_logps_1_true[y_obs.mask] = 0.

    states_step.compute_y_logp()
    y_logps_est = states_step.y_logp_vals

    assert_allclose(y_logps_est[0], y_logps_0_true)
    assert_allclose(y_logps_est[1], y_logps_1_true)

    mcmc_step.sample(2 * model_true['N_obs'])

    assert_hpd(mcmc_step.states_rv, states_true, alpha=0.1)


@pytest.mark.skip(reason="In progress...")
def test_states_trans_missing():
    """ TODO: Test for the case of missing observations when using
    both transition probability and state sequence step methods.
    """
    np.random.seed(2352523)

    model_true = simple_state_trans_model()

    trans_mat_true = model_true['trans_mat_rv'].value
    mu_true = model_true['mu'].value
    states_true = model_true['states_rv'].value
    y_true = model_true['y_rv'].value

    y_mask = np.random.randint(0, np.alen(y_true)-1,
                               int(0.2 * np.alen(y_true)))
    y_mask = np.in1d(np.arange(0, np.alen(y_true)), y_mask)
    y_obs = np.ma.masked_array(y_true, mask=y_mask)

    model_test = simple_state_trans_model(y_obs=y_obs)
    pymc.Normal
    mcmc_step = pymc.MCMC(model_test)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(PriorObsSampler, mcmc_step.y_rv)
    (y_step, ) = mcmc_step.step_method_dict[mcmc_step.y_rv]
    assert isinstance(y_step, PriorObsSampler)

    mcmc_step.assign_step_methods()

    #mcmc_step.use_step_method(HMMStatesStep, mcmc_step.states_rv)
    (states_step, ) = mcmc_step.step_method_dict[mcmc_step.states_rv]
    assert isinstance(states_step, HMMStatesStep)

    #mcmc_step.use_step_method(TransProbMatStep, mcmc_step.trans_mat_rv)
    (trans_mat_step, ) = mcmc_step.step_method_dict[mcmc_step.trans_mat_rv]
    assert isinstance(trans_mat_step, TransProbMatStep)

    mcmc_step.sample(1000)

    mcmc_step.y_rv.trace()[:].shape

    assert_hpd(mcmc_step.states_rv, states_true, alpha=0.1)
    assert_hpd(mcmc_step.trans_mat_rv, trans_mat_true, alpha=0.1)


def test_normal_normal_step():
    """ Test normal-normal step method.
    """
    np.random.seed(2352523)

    model_true = simple_norm_reg_model()

    X_matrices = model_true['X_matrices']
    betas_true = model_true['betas']
    trans_mat_true = model_true['trans_mat_obs']
    states_true = model_true['states_rv'].value
    mu_true = model_true['mu'].value
    y_obs = model_true['y_rv'].value

    betas = []
    for k, X_ in enumerate(X_matrices):
        beta_s = pymc.Normal('beta-{}'.format(k), 0, 1, size=X_.shape[1])
        betas += [beta_s]

    model_test = simple_norm_reg_model(X_matrices=X_matrices,
                                       betas=betas,
                                       y_obs=y_obs)

    mcmc_step = pymc.MCMC(model_test, verbose=2)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, model_test['states_rv'])

    (states_step,) = mcmc_step.step_method_dict[model_test['states_rv']]
    assert isinstance(states_step, HMMStatesStep)

    for b_ in mcmc_step.betas:
        mcmc_step.use_step_method(NormalNormalStep, b_)

    (beta1_step,) = mcmc_step.step_method_dict[mcmc_step.betas[0]]
    assert isinstance(beta1_step, NormalNormalStep)

    (beta2_step,) = mcmc_step.step_method_dict[mcmc_step.betas[1]]
    assert isinstance(beta2_step, NormalNormalStep)

    mcmc_step.sample(2000)

    assert_hpd(mcmc_step.states_rv, states_true, alpha=0.05)

    assert_hpd(mcmc_step.betas[0], betas_true[0], alpha=0.05)
    assert_hpd(mcmc_step.betas[1], betas_true[1], alpha=0.05)
