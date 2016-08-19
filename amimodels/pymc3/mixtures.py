import numpy as np

import pandas as pd

import patsy

import theano
import theano.typed_list
import theano.tensor as tt

import pymc3

from .deterministics import *
from .stochastics import *


def normal_mixture_reg_2(y_obs=None, X_mats=None,
                         dataframe=None, formulas=None):
    r""" Creates a PyMC3 mixture regression model using `theano.typed_list` and
    `theano.scan`.  Using `theano.scan` we can--among other things--potentially
    avoid explicit, static use of the observation vector length and, as a
    result, compile single, generic multi-use estimation functions.

    Parameters
    ==========
    dataframe: pandas.DataFrame
        DataFrame containing columns of variables referenced in `formulas`.
    formulas: collection of str
        Collection of Patsy strings that determine the design matrices
        of each mixture component/state.

    Returns
    =======
    A tuple containing the PyMC3 model object and the Theano shared variables
    for the observations (named 'y_obs') and design matrices (named 'X_{k}' for
    each `k` states).
    """
    if X_mats is not None and y_obs is not None:
        pass
    elif dataframe is not None and formulas is not None:
        X_mats = []

        for fm in formulas:
            y, X_ = patsy.dmatrices(fm, dataframe, return_type='dataframe')
            X_mats += [X_.values]

        y_obs = y.values.ravel()
    else:
        raise ValueError("Provide y_obs, X_mats or dataframe, formulas")

    y_T = theano.shared(y_obs, name="y_obs", borrow=True)

    X_mats_T = []
    for k, X in enumerate(X_mats):
        X_mats_T += [theano.shared(X, name="X_{}".format(k), borrow=True)]

    N_states = len(X_mats_T)

    with pymc3.Model() as mix_model:
        p_rv = pymc3.Dirichlet('p', np.ones(N_states))

        betas_T = []
        for i, X in enumerate(X_mats_T):
            mu_0 = np.concatenate([np.atleast_1d(np.percentile(y, (float(i)+1)/N_states * 100.)),
                                np.zeros(X.shape[1]-1)])
            sd_0 = 100.
            beta_ = pymc3.Normal('beta_{}'.format(i), mu=mu_0, sd=sd_0,
                                shape=(X.shape[1],))
            beta_ = theano.tensor.unbroadcast(beta_, 0)
            betas_T += [beta_]

        betas_list_T = theano.typed_list.make_list(betas_T)
        X_list_T = theano.typed_list.make_list(X_mats_T)

        states_rv = pymc3.Categorical('states', p=p_rv, shape=np.alen(y))

        def mu(t, s, bs_, Xs_):
            return Xs_[s][t].dot(bs_[s])

        mu_T, _ = theano.scan(fn=mu,
                              non_sequences=[betas_list_T, X_list_T],
                              sequences=[theano.tensor.arange(0, stop=np.alen(y_T)),
                                         states_rv])

        mu = pymc3.Deterministic('mu', mu_T)

        sigma_rv = pymc3.Uniform('sigma', 0, 20)
        y_rv = pymc3.Normal('y', mu=mu, sd=sigma_rv, observed=y_T)

    return mix_model, y_T, X_mats_T


def normal_mixture_reg(y_obs=None, X_mats=None, dataframe=None, formulas=None):
    """ Creates a PyMC3 mixture regression model from
    the columns of a pandas.DataFrame and formulas for each
    mixture component/state.

    Parameters
    ==========
    dataframe: pandas.DataFrame
        DataFrame containing columns of variables referenced in `formulas`.
    formulas: collection of str
        Collection of Patsy strings that determine the design matrices
        of each mixture component/state.

    Returns
    =======
    A tuple containing the PyMC3 model object and the Theano shared variables
    for the observations (named 'y_obs') and design matrices (named 'X_{k}' for
    each `k` states).
    """

    if X_mats is not None and y_obs is not None:
        pass
    elif dataframe is not None and formulas is not None:
        X_mats = []

        for fm in formulas:
            y, X_ = patsy.dmatrices(fm, dataframe, return_type='dataframe')
            X_mats += [X_.values]

        y_obs = y.values.ravel()
    else:
        raise ValueError("Provide y_obs, X_mats or dataframe, formulas")

    y_T = theano.shared(y_obs, name="y_obs", borrow=True)

    X_mats_T = []
    for k, X in enumerate(X_mats):
        X_mats_T += [theano.shared(X, name="X_{}".format(k), borrow=True)]

    N_states = len(X_mats_T)

    with pymc3.Model() as mix_model:

        p_rv = pymc3.Dirichlet('p', np.ones(N_states))

        ys_rv = []
        for k, X in enumerate(X_mats_T):
            N_features = X.get_value().shape[1]
            N_obs = np.alen(y_T.get_value())

            k_pct = np.percentile(y_T.get_value(),
                                  (float(k)+1)/N_states * 100.)

            mu_0 = np.concatenate([np.atleast_1d(k_pct),
                                   np.zeros(N_features-1)])

            #tau_theta_rv = pymc3.Uniform('tau_theta_{}'.format(k),
            #                             1e-5, 1.,
            #                             testval=1./N_obs)
            tau_theta_rv = pymc3.HalfCauchy('tau_theta_{}'.format(k),
                                            1.,
                                            testval=1.)

            sd_theta_rv = pymc3.HalfCauchy('sd_theta_{}'.format(k),
                                           1, shape=(N_features,),
                                           testval=np.ones(N_features))

            mu_theta_rv = 0

            theta_rv = pymc3.Normal('theta_{}'.format(k),
                                    mu=mu_theta_rv,
                                    sd=sd_theta_rv * tau_theta_rv,
                                    testval=mu_0,
                                    shape=(N_features,))

            sigma_y_upper = np.std(y_T.get_value())
            sigma_y_rv = pymc3.Uniform('sigma_y_{}'.format(k),
                                       1e-2, sigma_y_upper,
                                       testval=sigma_y_upper/N_states)

            #tau_y_rv = pymc3.Deterministic('tau_y_{}'.format(k),
            #                               1./tt.square(sigma_y_rv))

            mu_y_rv = pymc3.Deterministic('mu_y_{}'.format(k),
                                          X.dot(theta_rv))

            y_rv = pymc3.Normal('y_{}'.format(k),
                                mu=mu_y_rv, sd=sigma_y_rv,
                                shape=(N_obs,))
            ys_rv += [y_rv]


        y_obs_rv = FiniteMixture('y', p_rv, ys_rv, shape=(N_obs,), observed=y_T)

    return mix_model, y_T, X_mats_T


if __name__ == "__main__":

    #X_data = X_regmix

    formulas = ['usage ~ 1'] * 3 +\
               ['usage ~ 1 + C(temp.index.hour) * C(temp.index.weekday)']
               # ['usage ~ CDD + HDD + C(temp.index.weekday) + C(temp.index.month)']

    mix_logp_model, y_sT, X_mats_sT = normal_mixture_reg(X_data, formulas)

    with mix_logp_model:
        map_est_trans = pymc3.find_MAP()

    map_vars = untransform_vars(map_est_trans, mix_logp_model)

    logps = [np.log(map_vars['p'][i]) +
             logp_normal(X.dot(map_vars['theta_{}'.format(i)]), 1.,
                         y_sT).eval()
             for i, X in enumerate(X_mats_sT)]

    logps = np.vstack(logps).T

    states_map = logps.argmax(axis=1).astype(np.int)

    states_baseline = pd.DataFrame(states_map + 1,
                                   index=X_data.index,
                                   columns=[r'$S_t$'])

    mu_values = np.vstack((X.dot(map_vars['theta_{}'.format(k)]).eval()
                           for k, X in enumerate(X_mats_sT))).T
    mu_values = mu_values[np.arange(np.alen(states_map)), states_map]

    mu_baseline = pd.DataFrame(mu_values, index=X_data.index, columns=[r'$\mu_t$'])
