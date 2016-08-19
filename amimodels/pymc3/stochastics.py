import numpy as np

from scipy import stats

import theano.tensor as tt

import pymc3
from pymc3.math import LogSumExp
from pymc3.distributions.distribution import generate_samples, draw_values


class FiniteMixture(pymc3.Distribution):
    r""" A finite mixture distribution.
    This object holds a list of pymc3.Distribution objects
    as its components/states.  In this scenario, the caller
    can specify all the details and this abstraction handles
    only small logp and random functionality.

    XXX: Not sure about this approach.  It seems difficult
    to work with since we can't make the Distributions
    in our list observed, which would probably be easier.
    Need to look into this more.
    """

    #def __new__(cls, name, *args, **kwargs):
    #    """ Just a hack of the original __new__; passes
    #    observed data to the __init__ constructor so that
    #    we can get concrete shape info.
    #
    #    This whole idea is really about automatic shape
    #    determination/symbolic distribution shapes, which
    #    is being handled for pymc3 elsewhere (see my PR in Github).
    #    """
    #    try:
    #        model = pymc3.Model.get_context()
    #    except TypeError:
    #        raise TypeError("No model on context stack. "
    #                        "Add a 'with model:' block")

    #    if isinstance(name, str):
    #        data = kwargs.pop('observed', None)
    #        dist = cls.dist(data, *args, **kwargs)
    #        return model.Var(name, dist, data)
    #    elif name is None:
    #        return object.__new__(cls)  # for pickle
    #    else:
    #        raise TypeError("needed name or None but got: " + name)

    def __init__(self, data, pis, ys, *args, **kwargs):
        r"""
        Parameters
        ==========
        pis: `pymc3.Dirichlet`s
            Mixture probabilities/weights.
        ys: list of `pymc3.Distribution`.
            Distributions in the mixture.
        """
        self.pis = pis
        self.ys = ys

        dtype = args.get('dtype', ys[0].dtype)
        shape = ys[0].shape
        testval = getattr(ys[0].tag, 'test_value', None)

        #self.data = data
        #self.dtype = dtype
        #self.type = ys[0].type
        #self.testval = testval
        #self.defaults = None
        #self.transform = None
        super(FiniteMixture, self).__init__(shape, dtype, testval, *args,
                                            **kwargs)

    @property
    def shape(self):
        r"""
        This is a feature we probably want to see in pymc3.Distribution
        in general.  Simply put, we get concrete shape values when the
        shape of this Distribution is based on a shared or constant Theano variable.
        """
        if self.data is not None:
            gval = getattr(self.data, 'get_value', self.data)
            return np.shape(gval)
        else:
            return self.ys[0].distribution.shape

    def random(self, point=None, size=None):
        # FIXME: Unfinished.
        pis_smpl = draw_values(self.pis, point=point)
        ys_smpl = draw_values(self.ys, point=point)

        def _random(pis, ys, size=None):
            which_k = stats.dirichlet.rvs(pis,
                                          None if size == pis.shape else size)
            return ys[tt.arange(self.shape), which_k]

        samples = generate_samples(_random, pis_smpl, ys_smpl,
                                   dist_shape=self.shape,
                                   size=size)
        return samples

    def logp(self, value):
        #logps = [tt.log(self.pis[k]) + logp_normal(mu, tau, value)
        #         for k, (mu, tau) in enumerate(zip(self.mus, self.taus))]
        logps = [tt.log(self.pis[k]) + y.distribution.logp(value)
                 for k, y in enumerate(self.ys)]
        logp_states = tt.stacklists(logps)
        return tt.sum(LogSumExp(logp_states, axis=0))
