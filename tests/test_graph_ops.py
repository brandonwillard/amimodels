from __future__ import division

import sys

import pandas as pd
import numpy as np
import pymc

import pytest
from numpy.testing import assert_array_equal

from amimodels.graph import parse_node_partitions, normal_node_update
from amimodels.normal_hmm import make_normal_hmm

if hasattr(sys, '_called_from_test'):
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
    )


@pytest.fixture(params=[np.array([False]*10),
                        np.array([True]*5 + [False]*5),
                        np.array([False]*5 + [True] + [False]*4),
                        ])
def hmm_masked(request):

    np.random.seed(2352523)
    N_obs = np.alen(request.param)
    y_obs = pd.Series(np.arange(N_obs))
    X_matrices = [pd.DataFrame(np.ones((N_obs, 1))),
                  pd.DataFrame(np.ones((N_obs, 2)))]

    beta_1_tau_rv = pymc.Gamma('beta-1-tau', 1/2., 1/2., value=1e-2)
    beta_2_tau_rv = pymc.Gamma('beta-2-tau', 1/2., 1/2., size=2,
                               value=np.array([2e-2, 3e-2]))

    beta_1_rv = pymc.Normal('beta-1', 1, beta_1_tau_rv, value=1)
    beta_2_rv = pymc.Normal('beta-2', np.array([2, 2]),
                            beta_2_tau_rv, size=2,
                            value=np.array([2, 2]))
    betas = [beta_1_rv, beta_2_rv]

    from amimodels.deterministics import HMMLinearCombination
    states_rv = pymc.Binomial('states', 1, 0.5, size=np.alen(y_obs))
    mu_rv = HMMLinearCombination('mu', X_matrices,
                                 [beta_1_rv, beta_2_rv], states_rv)

    y_tau_rv = pymc.Gamma('y-tau', 1/2., 1/2., value=1.)

    taus = [y_tau_rv, y_tau_rv]

    y_obs_val = np.ma.masked_array(y_obs, request.param, dtype=np.float)

    y_rv = pymc.Normal('y', mu_rv, y_tau_rv,
                       value=y_obs_val, observed=True)

    return y_rv, states_rv, betas, taus, X_matrices


def test_collapse(hmm_masked):
    """ Test that simple normal observation distribution marginalization
    on a normally distributed mean parameter works with masked/missing
    observations.  This test uses a basic HMM state sequence.
    """

    # DEBUG:
    #from collections import namedtuple
    #Request = namedtuple('Request', ['param'])
    #request = Request(np.array([False]*10))
    #request = Request(np.array([False]*5 + [True] + [False]*4))
    #y_rv, states_rv, betas, X_matrices = hmm_masked(request)

    y_rv, states_rv, betas, taus, X_matrices = hmm_masked

    updates = parse_node_partitions(y_rv)

    for k, node_k in enumerate(updates):
        mu_k = node_k.parents['mu']
        tau_k = node_k.parents['tau']
        which_k = pymc.utils.value(states_rv) == k

        assert_array_equal(pymc.utils.value(mu_k),
                           pymc.utils.value(y_rv.parents['mu'])[which_k])

        assert np.array_equiv(pymc.utils.value(tau_k),
                              pymc.utils.value(taus[k]))

        node_k_mask = pymc.utils.value(getattr(node_k, '_mask', None))
        assert_array_equal(np.ma.masked_array(pymc.utils.value(node_k),
                                              node_k_mask),
                           pymc.utils.value(y_rv)[which_k])

    normal_node_update(y_rv, updates)

    y_marginal, = y_rv.marginals

    assert y_marginal.observed == y_rv.observed
    assert_array_equal(y_marginal.value, y_rv.value)
    assert_array_equal(y_marginal.mask, y_rv.mask)

    true_marg_mu = np.fromiter((np.dot(X_matrices[s_].iloc[t_],
                                       betas[s_].parents['mu']).squeeze()
                                for t_, s_ in enumerate(pymc.utils.value(states_rv))),
                               dtype=np.float)
    assert_array_equal(pymc.utils.value(y_marginal.parents['mu']),
                       true_marg_mu)

    # This is called 'lam' when the marginal is t-distributed.
    true_marg_tau = np.fromiter((
        np.reciprocal(1 / pymc.utils.value(y_rv.parents['tau']) +
                      np.sum(np.square(X_matrices[s_].iloc[t_]) /
                             pymc.utils.value(betas[s_].parents['tau'])))
        for t_, s_ in enumerate(pymc.utils.value(states_rv))),
        dtype=np.float)

    assert_array_equal(pymc.utils.value(y_marginal.parents['lam']),
                       true_marg_tau)

    for k, marg_part in enumerate(y_marginal.partitions):
        mu_k = marg_part.parents['mu']
        lam_k = marg_part.parents['lam']
        which_k = pymc.utils.value(states_rv) == k
        assert_array_equal(pymc.utils.value(mu_k),
                           true_marg_mu[which_k])
        assert_array_equal(pymc.utils.value(lam_k),
                           true_marg_tau[which_k])


@pytest.mark.skip(reason="In progress...")
def test_collapse_norm_hmm():
    """ Test that simple normal observation distribution marginalization
    on a normally distributed mean parameter works.
    """

    np.random.seed(2352523)
    N_obs = 20
    #y_obs = pd.Series(np.arange(N_obs))
    y_obs = pd.Series(np.zeros(N_obs))
    X_matrices = [pd.DataFrame(np.ones((N_obs, 1))),
                  pd.DataFrame(np.ones((N_obs, 2)))]

    norm_hmm = make_normal_hmm(y_obs, X_matrices)
    node, = norm_hmm.observed_stochastics

    #from amimodels.graph import *
    #from amimodels.deterministics import *
    updates = parse_node_partitions(node)

    normal_node_update(node, updates)

    y_marginal, = norm_hmm.y_rv.marginals

    # Get rid of the dependencies on this node
    # and create a new model with the marginal
    # replacing the un-marginalized.
    #obs_node.parents.detach_extended_parents()
    y_marginal.parents['mu'].keep_trace = True
    new_nodes = set((y_marginal, y_marginal.parents['mu'])) |\
        y_marginal.extended_parents

    #_ = [n_ for n_ in new_nodes]
    new_norm_hmm = pymc.Model(new_nodes)

    mcmc_step = pymc.MCMC(new_norm_hmm.variables)

    mcmc_step.assign_step_methods()
    mcmc_step.step_method_dict

    mcmc_step.sample(200)

    mcmc_step.db.trace_names

    new_norm_hmm.get_node('states').stats()
    new_norm_hmm.get_node('mu_y_marg').stats()
    #y_marginal.parents['mu'].trace()


@pytest.mark.skip(reason="In progress...")
def test_collapse_binomial():
    """ Test for a binomial sequence.
    TODO: Finish
    """

    np.random.seed(2352523)
    N_obs = 20
    y_obs = pd.DataFrame(np.ones(N_obs))
    X_matrices = [pd.DataFrame(np.ones((N_obs, 1))),
                  pd.DataFrame(np.ones((N_obs, 2)))]

    from amimodels.deterministics import HMMLinearCombination
    from amimodels.stochastics import HMMStateSeq

    beta_1_tau_rv = pymc.Gamma('beta-1-tau', 1/2., 1/2.)
    beta_2_tau_rv = pymc.Gamma('beta-2-tau', 1/2., 1/2., size=2)
    #beta_1_tau_rv = pymc.HalfCauchy('beta-1-tau', 0, 1)
    #beta_2_tau_rv = pymc.HalfCauchy('beta-2-tau', 0, 1, size=2)
    #beta_1_tau_rv = 1.
    #beta_2_tau_rv = np.tile(2., (2,))
    beta_1_rv = pymc.Normal('beta-1', 1, beta_1_tau_rv)
    beta_2_rv = pymc.Normal('beta-2', 0, beta_2_tau_rv, size=2)

    states_rv = pymc.Binomial('states', 1, 0.5, size=np.alen(y_obs))
    mu_rv = HMMLinearCombination('mu', X_matrices,
                                 [beta_1_rv, beta_2_rv], states_rv)
    #mu_rv = pymc.LinearCombination('mu', [X_matrices[1]], [beta_2_rv])
    #mu_rv = 0.

    y_tau_rv = pymc.Gamma('y-tau', 1/2., 1/2.)
    #y_tau_rv = pymc.HalfCauchy('y-tau', 0, 1)
    #y_tau_rv = 1.

    y_obs_val = np.asarray(y_obs).ravel()
    #y_obs.iloc[5] = None
    #y_obs.iloc[12] = None
    #y_obs_val = np.ma.masked_invalid(y_obs)

    y_rv = pymc.Normal('y', mu_rv, y_tau_rv,
                       value=y_obs_val, observed=True)
    #y_rv = pymc.Normal('y', mu_rv, y_tau_rv)

    node = y_rv
    updates = parse_node_partitions(node)
    normal_node_update(node, updates)

    y_marg, = y_rv.marginals

    assert_array_equal(y_marg.value, y_rv.value)
    assert_array_equal(y_marg.parents['mu'].value,
                       np.array([1, 0])[states_rv.value])
    #np.reciprocal(1/y_tau_rv.value + np.transpose(X_matrices[0] /
    #              beta_1_rv.parents['tau'].value).dot(X_matrices[0]))
    #assert_array_equal(y_marg.parents['tau'].value,
    #                   np.array([1, 0])[states_rv.value])


@pytest.mark.skip(reason="In progress...")
def test_multiple_partitions():
    """ Test that we can process multiple partitions of a node.
    TODO, FIXME: We currently can't handle this.
    """

    np.random.seed(2352523)
    N_obs = 20
    y_obs = pd.Series(np.arange(N_obs))
    X_matrices = [pd.DataFrame(np.ones((N_obs, 1))),
                  pd.DataFrame(np.ones((N_obs, 2)))]

    norm_hmm = make_normal_hmm(y_obs, X_matrices)

    from amimodels.stochastics import HMMStateSeq
    tau_states_rv = HMMStateSeq("tau_states",
                                np.asarray([[0.9], [.1]]), N_obs)

    from amimodels.deterministics import NumpyChoose
    V_inv = NumpyChoose(tau_states_rv, norm_hmm.V_inv_list,
                        out=None)

    old_obs_node, = norm_hmm.observed_stochastics

    node = pymc.Normal('y_rv', old_obs_node.parents['mu'],
                       V_inv, value=old_obs_node.value,
                       observed=True)

    #from amimodels.graph import *
    #from amimodels.deterministics import *

    updates = parse_node_partitions(node)


if __name__ == "__main__":
    import pytest
    pytest.main(["-x", "-s", "--pdbcls=IPython.core.debugger:Pdb",
                "tests/test_graph_ops.py::test_collapse"])
