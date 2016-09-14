from __future__ import division

import sys

import numpy as np
import pymc

import pytest
from amimodels.testing import assert_hpd

from amimodels.stochastics import (dvalue_class, HMMStateSeq)

from amimodels.normal_hmm import NormalHMMProcess
from amimodels.step_methods import HMMStatesStep

if hasattr(sys, '_called_from_test'):
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


@pytest.mark.skip(reason="In progress...")
@slow
def test_hmmstateseq_mcmc(process_2_state_trans,
                          mcmc_iters=2000):
    """Simulate observations for a mixture of normals with a fixed/known
    transition probability matrix and the HMMStateSeq stochastic.
    Check that the stochastic's terms are estimated properly.
    """
    trans_mat = process_2_state_trans.trans_mat
    N_obs = process_2_state_trans.N_obs
    test_states = HMMStateSeq('states', trans_mat, N_obs)

    @pymc.deterministic(trace=True, plot=False)
    def mu(states_=test_states):
        return np.asarray([-1, 1])[states_]

    states_obs = test_states.value
    y_obs_rv = pymc.Normal('y_obs', mu, 100)
    y_obs = y_obs_rv.random()

    test_y = pymc.Normal('y', mu, 100, value=y_obs, observed=True)

    norm_hmm = {'states': test_states, 'y': test_y}

    mcmc_step = pymc.MCMC(norm_hmm.variables)
    mcmc_step.draw_from_prior()

    mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)

    mcmc_step.sample(mcmc_iters,
                     burn=mcmc_iters//2)

    assert_hpd(test_states, states_obs)


def test_dvalue_class():

    tv1_dval = pymc.Lambda('', lambda x_=1.2: x_)
    tv1_dval_inst = dvalue_class(pymc.Normal,
                                 'tv1-d', 0, 1,
                                 value=tv1_dval)

    assert isinstance(tv1_dval_inst, pymc.Normal)
    assert isinstance(tv1_dval_inst, pymc.Stochastic)

    assert tv1_dval_inst._dvalue == tv1_dval
    assert tv1_dval_inst.value == 1.2
    assert tv1_dval_inst._value == 1.2

    tv1_dval_inst.value = 1.5

    assert tv1_dval_inst._dvalue == 1.5
    assert tv1_dval_inst.value == 1.5
    assert tv1_dval_inst._value == 1.5

    tv1 = pymc.Normal('tv1', 0, 1)
    tv1_dval_inst._dvalue = tv1

    assert tv1_dval_inst._dvalue == tv1
    assert tv1_dval_inst.value == tv1.value
    assert tv1_dval_inst._value == tv1.value

    tv1.random()

    assert tv1_dval_inst._dvalue == tv1
    assert tv1_dval_inst.value == tv1.value
    assert tv1_dval_inst._value == tv1.value

    tv2 = pymc.Normal('tv2', 0, 1)
    tv2_dval_inst = dvalue_class(pymc.Normal,
                                 'tv2-d', 0, 1,
                                 value=tv2,
                                 observed=True)

    assert tv2_dval_inst.observed

    assert tv2_dval_inst._dvalue == tv2
    assert tv2_dval_inst.value == tv2.value
    assert tv2_dval_inst._value == tv2.value

    try:
        tv2_dval_inst.value = tv2
    except AttributeError:
        pass
    else:
        assert False, "Set `value` of observed didn't throw"

