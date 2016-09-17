r"""
This module provides PyMC Deterministic objects custom built
for hidden Markov models.

.. moduleauthor:: Brandon T. Willard
"""
import collections

import numpy as np
import pymc

from pymc.NumpyDeterministics import deterministic_from_funcs


NumpyTake = deterministic_from_funcs("take", np.take)
NumpyChoose = deterministic_from_funcs("choose", np.choose)
NumpyPut = deterministic_from_funcs("put", np.put)
NumpyStack = deterministic_from_funcs("stack", np.stack)
NumpyHstack = deterministic_from_funcs("hstack", np.hstack)
NumpyBroadcastArrays = deterministic_from_funcs("broadcast_arrays",
                                                np.broadcast_arrays)
NumpyMultidot = deterministic_from_funcs("multi_dot", np.linalg.multi_dot)


class KIndex(pymc.Deterministic):
    r""" A deterministic that dynamically tracks which time/first axis
    indices correspond to a given integer value.

    It keeps a static record of the indices it creates inside of
    any pymc.Node that is being indexed.  That means we don't create
    duplicates for the same k-index.

    XXX: Can't trace these Deterministics, since they change shape.
    """
    def __new__(cls, k, states, *args, **kwds):
        if isinstance(states, pymc.Node):
            k_indices = getattr(states, 'k_indices', {})
            k_index = k_indices.get(k, None)

            if k_index is not None:
                return k_index

        return super(KIndex, cls).__new__(cls, k, states,
                                          *args, **kwds)

    def __init__(self, k, states, *args, **kwds):
        r"""
        Parameters
        ==========
        k: int
            State to track.
        states: pymc.Node, ndarray of int
            State sequence in which `k` is a possible state.
        """
        if not np.issubdtype(getattr(k, 'dtype', type(k)),
                             np.integer):
            raise ValueError("k must be an int: type(k)={}".format(type(k)))

        if not np.issubdtype(getattr(states, 'dtype', type(k)),
                             np.integer):
            raise ValueError(("states must be an int:"
                             "type(states)={}".format(type(states))))

        parents = {'k': k, 'states': states}

        def k_idx(k, states):
            return np.flatnonzero(k == states)

        name = "which_{}_eq_{}".format(states.__name__, k)

        super(KIndex, self).__init__(eval=k_idx, doc=self.__doc__,
                                     name=name, parents=parents,
                                     trace=False, dtype=np.uint,
                                     *args, **kwds)

        if isinstance(states, pymc.Node):
            k_indices = getattr(states, 'k_indices', {})
            k_indices.update({k: self})
            states.k_indices = k_indices


indexers = (NumpyTake, NumpyChoose, KIndex)


class HMMLinearCombination(pymc.Deterministic):
    r""" A deterministic that represents

    .. math::

        \mu_t = x_t^{(S_t)\top} \beta^{(S_t)}

    for a state sequence :math:`\{S_t\}_{t=1}^T` with :math:`S_t \in \{1,\dots,K\}`,
    rows of design matrices :math:`x^{(k)}_t`, and covariate vectors
    :math:`\beta^{(k)}`.

    This deterministic organizes, separates and tracks the aforementioned
    :math:`\mu_t` in pieces designated by the current state sequence, :math:`S_t`.
    Specficially, it tracks a set containing the following sets for
    :math:`k \in \{1,\dots,K\}`

    .. math::
        \left\{ \mu^{(k)} = \tilde{X}^{(k)} \beta^{(k)},
           \tilde{X}^{(k)} = X_{\mathcal{T}^{(k)}},
           \mathcal{T}^{(k)} = \{t : S_t = k\}
        \right\}


    """

    def __init__(self, name, X_matrices, betas, states, *args, **kwds):
        r"""
        Parameters
        ==========
        X_matrices: list of np.ndarrays
            Matrices for each possible value of :math:`S_t` with rows
            corresponding to :math:`x_t` in the product
            :math:`x_t^\top \beta^{(S_t)}`.
        betas: list of np.ndarrays or pymc.Stochastics
            Vectors corresponding to each :math:`beta^{(S_t)}` in the
            product :math:`x_t^\top \beta^{(S_t)}`.
        states: np.ndarray of int
            Vector of the state sequence :math:`S_t`.
        """

        self.N_states = np.alen(X_matrices)

        if not self.N_states == np.alen(betas):
            raise ValueError("len(X_matrices) must equal len(betas)")

        if not np.count_nonzero(np.diff([np.alen(X_)
                                         for X_ in X_matrices])) == 0:
            raise ValueError("X_matrices must all have equal first dimension")

        self.N_obs = np.alen(X_matrices[0])

        if not self.N_obs == np.alen(states.value):
            raise ValueError("states do not match X_matrices dimensions")

        if not all([np.shape(X_)[-1] == np.alen(getattr(b_, 'value', b_))
                    for X_, b_ in zip(X_matrices, betas)]):
            raise ValueError("X_matrices and betas dimensions don't match")

        # TODO: Change these to indices instead of boolean masks
        self.states = states
        self.which_k = tuple()
        self.X_k_matrices = tuple()
        self.k_prods = tuple()
        self.beta_obs_idx = dict()

        for k in xrange(self.N_states):

            w_k = KIndex(k, states)
            self.which_k += (w_k,)

            X_k_mat = NumpyTake(np.asarray(X_matrices[k]),
                                w_k, axis=0)
            self.X_k_matrices += (X_k_mat,)

            this_beta = betas[k]
            k_p = pymc.LinearCombination("mu_{}".format(k),
                                         (X_k_mat,),
                                         (this_beta,),
                                         trace=False)
            self.k_prods += (k_p,)

            if isinstance(this_beta, collections.Hashable):
                self.beta_obs_idx[this_beta] = w_k

        def eval_fun(which_k, k_prods):
            res = np.empty(self.N_obs, dtype=np.float)
            for idx, k_p in zip(which_k, k_prods):
                res[idx] = k_p
            return res

        parents = {'which_k': self.which_k, 'k_prods': self.k_prods}
        super(HMMLinearCombination, self).__init__(eval=eval_fun,
                                                   doc=self.__doc__,
                                                   name=name,
                                                   parents=parents,
                                                   *args, **kwds)


def get_indexed_items(node):
    res = None, None
    if isinstance(node, NumpyTake):
        idx = node.parents['indices']
        col = node.parents['a']
        res = (idx, col)
    elif isinstance(node, NumpyChoose):
        idx = node.parents['a']
        col = node.parents['choices']
        res = (idx, col)
    elif isinstance(node, HMMLinearCombination):
        idx = node.states
        col = node.k_prods
        res = (idx, col)
    elif "index" in node.parents.keys():
        idx = node.parents['index']
        col = node.parents['self']
        res = (idx, col)

    return res


def parse_prod_index(node):
    """ FIXME
    """

    if isinstance(node, HMMLinearCombination):
        mu_s = node.k_prods
        k_idx_s = node.which_k

    elif isinstance(node, NumpyHstack):
        # Collapse a chain of products and indexing.
        mu_s = node.parents['tup']

        for m_ in mu_s:
            n_ = m_
            a_ = 1
            m_idx = None
            while n_ is not None:
                b_ = None
                if isinstance(n_, pymc.LinearCombination):
                    # Take the left term in the product,
                    # since any indexing on the first dimension of this
                    # is what we want.

                    # FIXME: Perhaps a little restrictive taking only the
                    # first elements.
                    a_ *= n_.x[0]
                    b_, = n_.y

                    # TODO
                    #if isinstance(b_, pymc.Deterministic): and\
                    #        np.shape(b_)[-1] == :
                    #    n_ = b_

                elif isinstance(n_, pymc.Deterministic) and\
                    "mul_" in n_.__name__ and\
                        ('a' in n_.parents.keys() and
                         'b' in n_.parents.keys()):
                    a_ = n_.parents['a']
                    b_ = n_.parents['b']
                else:
                    continue

                n_idx = None
                if isinstance(a_, indexers):
                    n_idx = getattr(a_.parents, 'indices', None)
                elif isinstance(n_, pymc.Node):
                    n_idx = getattr(n_.parents, 'index', None)

                if n_idx is not None:
                    # TODO
                    pass

                n_ = b_

            k_idx_s += (m_idx,)

    elif isinstance(node, pymc.LinearCombination):
        mu_s = node
        # TODO
    else:
        return None

    return zip(mu_s, k_idx_s)
