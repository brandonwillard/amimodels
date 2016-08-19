r"""
This module provides PyMC3 Deterministic objects custom built
for hidden Markov models.

.. moduleauthor:: Brandon T. Willard
"""
import numpy as np

import theano
import theano.tensor as tt

import pymc3


def untransform_vars(map_est, model):
    """ Un-apply random variable transforms.

    Parameters
    ==========
    map_est: a dict
        A dict containing transformed variables.
    model: a pymc3.Model
        The pymc3 model object that contains transform info.

    Returns
    =======
    A dict of un-transformed variables.
    """
    from warnings import warn

    map_values = {}
    for var_name, val in map_est.items():
        var = model.named_vars.get(var_name, None)
        if var is not None:
            var_trans = getattr(var.distribution, 'transform_used', None)
            if var_trans is not None:
                var_name_new = var_name.replace("_" + var_trans.name, '')
                var_val = var_trans.backward(val).eval()
                map_values[var_name_new] = var_val
            else:
                map_values[var_name] = val
        else:
            warn("unmapped variable")

    return map_values


def logp_normal(mu, tau, value):
    r""" Normal log likelihood function.
    """
    return pymc3.dist_math.bound((-tau * (value - mu)**2 +
                                  tt.log(tau / np.pi / 2.)) / 2.,
                                 tau > 0)
