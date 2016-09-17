import numpy as np
import pymc

from .hmm_utils import compute_steady_state
from .deterministics import KIndex


def dvalue_class(cls, *args, **kwargs):
    """ Creates a new class for a pymc.Node in which
    `Node.value` is a property that can evaluate another
    pymc.Node, i.e. a "dynamic" value class.
    Simply put, `Node.value` calls another `Node.value`.

    Parameters
    ==========
    cls: type
        The class we want to extend/re-create.
    *args, **kwargs:
        Arguments that match the constructor of ``cls``.
        A `value` keyword can now be an object which itself
        has a `value` property.  It will then get evaluated
        dynamically whenever we get `value` for the new class.

    Returns
    =======
    instance of ``cls`` with pass-thru `value` property
    returning ``dvalue.value``.
    """
    dcls = dict(cls.__dict__)
    dcls.update({'_dvalue': None})

    cls_new = type("{}_dvalue".format(cls.__name__),
                   (cls, object), dcls)

    value_prop = property(lambda self:
                          getattr(self._dvalue, 'value', self._dvalue),
                          lambda self, value:
                          setattr(self, '_dvalue', value),
                          lambda *x, **y: None)
    setattr(cls_new, "_value", value_prop)

    init_value = kwargs.pop('value', None)
    init_observed = kwargs.pop('observed', False)

    res = cls_new(*args, **kwargs)

    if init_value is not None:
        setattr(res, '_dvalue', init_value)
    if init_observed is not None:
        setattr(res, '_observed', init_observed)

    return res


def trans_mat_logp(value, alpha_trans):
    """ Computes the log probability of a transition probability
    matrix for the given Dirichlet parameters.

    Parameters
    ==========
    value: ndarray
        The observations.

    alpha_trans: ndarray
        The Dirichlet parameters.

    Returns
    =======
    The log probability.
    """
    logp = 0.
    for s, alpha_row in enumerate(alpha_trans):
        logp += pymc.dirichlet_like(value[s], alpha_row)
    return logp


def trans_mat_random(alpha_trans):
    """ Sample a shape (K, K-1) transition probability matrix
    with K-many states given Dirichlet parameters for each row.

    Parameters
    ==========
    alpha_trans: ndarray
        Dirichlet parameters for each row.

    Returns
    =======
    A ndarray of the sampled transition probability matrix (without the
    last column).
    """
    trans_mat_smpl = np.empty((alpha_trans.shape[0],
                               alpha_trans.shape[1]-1), dtype=np.float)
    for s, alpha_row in enumerate(alpha_trans):
        trans_mat_smpl[s] = pymc.rdirichlet(alpha_row, size=1)
    return trans_mat_smpl


class TransProbMatrix(pymc.Stochastic):
    r""" A stochastic that represents an HMM's transition probability
    matrix with rows given by

    .. math::

        \pi^{(k)} \sim \operatorname{Dir}(\alpha^{(k)})
        \;,

    for :math:`k \in \{1, \dots, K\}`.

    This object technically works with the :math:`K-1`-many **columns** of
    transition probabilities, and each row is represented a Dirichlet
    distribution (in the :math:`K-1` independent
    terms.

    Parameters
    ==========

    alpha_trans: ndarray
        Dirichlet parameters for each row of the transition probability matrix.

    value: ndarray of int
        Initial value.

    See Also
    ========
    pymc.PyMCObjects : Stochastic

    """

    def __init__(self, name, alpha_trans, *args, **kwargs):

        if len(alpha_trans.shape) != 2 or \
                alpha_trans.shape[0] != alpha_trans.shape[1]:
            raise ValueError("alpha_trans must be a square matrix")

        # TODO: Might be good to set the shape (K, K-1) manually and/or
        # perform some checks.
        super(TransProbMatrix, self).__init__(logp=trans_mat_logp,
                                              doc=TransProbMatrix.__doc__,
                                              name=name,
                                              parents={'alpha_trans':
                                                       alpha_trans},
                                              random=trans_mat_random,
                                              dtype=np.dtype('float'),
                                              *args,
                                              **kwargs)


def states_logp(value, trans_mat, N_obs, p0):
    """ Computes log likelihood of states in an HMM.

    Parameters
    ==========

    value: ndarray of int
        Array of discrete states numbers/indices/labels

    trans_mat: ndarray
        A transition probability matrix for K states with shape
        (K, K-1), i.e. the last column omitted.

    N_obs: int
        Number of observations.

    p0: ndarray
        Initial state probabilities.  If `None`, the steady state
        is computed and used.

    Returns
    =======

    float value of the log likelihood

    """

    if np.any(np.logical_or(value < 0, value >= trans_mat.shape[0])):
        return -np.inf

    P = np.column_stack((trans_mat, 1. - trans_mat.sum(axis=1)))
    p = p0
    if p0 is None:
        p = compute_steady_state(trans_mat)

    logp = pymc.categorical_like(value[0], p)
    for t in xrange(1, len(value)):
        logp += pymc.categorical_like(value[t], P[int(value[t-1])])

    return logp


def states_random(trans_mat, N_obs, p0):
    """ Samples states from an HMM.

    Parameters
    ==========

    trans_mat: ndarray
        A transition probability matrix for K-many states with
        shape (K, K-1).

    N_obs: int
        Number of observations.

    p0: ndarray
        Initial state probabilities.  If `None`, the steady state
        is computed and used.

    Returns
    =======

    A ndarray of length N_obs containing sampled state numbers/indices/labels.

    """

    P = np.column_stack((trans_mat, 1. - trans_mat.sum(axis=1)))
    p = p0
    if p is None:
        p = compute_steady_state(trans_mat)

    states = np.empty(N_obs, dtype=np.uint8)

    states[0] = pymc.rcategorical(p)
    for i in range(1, N_obs):
        states[i] = pymc.rcategorical(P[int(states[i-1])])

    return states


class HMMStateSeq(pymc.Stochastic):
    """ A stochastic that represents an HMM's state
    process :math:`\{S_t\}_{t=1}^T`.  It's basically the
    distribution of a sequence of Categorical distributions connected
    by a discrete Markov transition probability matrix.

    Use the step methods made specifically for this distribution;
    otherwise, the default Metropolis samplers will likely perform
    too poorly.

    See Also
    ========
    pymc.PyMCObjects : Stochastic

    """

    def __init__(self, name, trans_mat, N_obs, p0=None, *args, **kwargs):
        """
        Parameters
        ==========

        trans_mat: ndarray
            A transition probability matrix for K-many states with shape
            (K, K-1).

        N_obs: int
            Number of observations.

        p0: ndarray
            Initial state probabilities.  If `None`, the steady state
            is computed and used.

        value: ndarray (int)
            Initial value array of discrete states numbers/indices/labels.
        """

        self.N_states = trans_mat.shape[0]
        # Legacy?
        self.K = self.N_states

        if not (len(trans_mat.shape) == 2 and
                trans_mat.shape[1] == self.N_states - 1):
            raise ValueError(("trans_mat must have ndim == 2 and "
                              "shape (K, K-1)"))

        super(HMMStateSeq, self).__init__(logp=states_logp,
                                          doc=HMMStateSeq.__doc__,
                                          name=name,
                                          parents={'trans_mat': trans_mat,
                                                   'N_obs': N_obs,
                                                   'p0': p0},
                                          random=states_random,
                                          dtype=np.int,
                                          *args,
                                          **kwargs)

        # Create Deterministics that effectively give `range(N_obs)[k == self]`
        # for each k.  KIndex's constructor should populate `self.k_indices`.
        for k in xrange(self.N_states):
            KIndex(k, self)
