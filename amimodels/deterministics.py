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
NumpyAlen = deterministic_from_funcs("alen", np.alen)
NumpySum = pymc.NumpyDeterministics.sum

# XXX: See below for likely similar issues.
NumpyBroadcastArrays = deterministic_from_funcs("broadcast_arrays",
                                                np.broadcast_arrays)


class NumpyBroadcastTo(pymc.Deterministic):
    """
    Can't do this:

        NumpyBroadcastTo = deterministic_from_funcs("broadcast_to", np.broadcast_to)

    Weird stuff with array memory and/or "contiguity" using broadcast_to as
    a Deterministic then taking a dot product later.  A quick fix involves
    creating a copy of the broadcasted array instead of a view.
    """

    def __init__(self, array, shape, subok=False, **kwds):
        array = np.asarray(array)

        def broadcast_to_(array=array, shape=shape, subok=subok):
            return np.array(np.broadcast_to(array, shape,
                                            subok=subok),
                            dtype=array.dtype)

        if isinstance(array, pymc.Node):
            array_name = array.__name__
        else:
            array_name = "array"

        name = "broadcast_to({}, {})".format(array_name, shape)
        super(NumpyBroadcastTo, self).__init__(eval=broadcast_to_,
                                               doc=np.broadcast_to.__doc__,
                                               name=name,
                                               parents={'array': array,
                                                        'shape': shape,
                                                        'subok': subok},
                                               trace=False,
                                               dtype=array.dtype,
                                               **kwds)

NumpyMultidot = deterministic_from_funcs("multi_dot", np.linalg.multi_dot)


class MergeIndexed(pymc.Deterministic):
    r""" A deterministic that merges multiple indexed arrays
    into a single array.

    XXX: Can't trace these Deterministics when they change shape.
    """
    def __init__(self, arrays, indices, *args, **kwds):
        r"""
        Parameters
        ==========
        arrays: list/tuple of pymc.Node or numpy.ndarray
            Array-like objects to merge
        indices: list/tuple of pymc.Node or numpy.ndarray
            Indices for the arrays in ``arrays``.
        """
        if not all(np.issubdtype(getattr(i_, 'dtype', type(i_)),
                                 np.integer) for i_ in indices):
            raise ValueError("indices must be int types")

        parents = {'values_': arrays, 'indices_': indices}

        def merge_by_indices(values_=arrays,
                             indices_=indices):
            m_values = np.hstack(values_)
            m_indices = np.hstack(indices_)
            res = np.empty(np.shape(m_indices))
            np.put(res, m_indices, m_values)
            return res

        name = "Merge({},{})".format(getattr(arrays, '__name__', 'arrays'),
                                     getattr(indices, '__name__', 'indices'))

        super(MergeIndexed, self).__init__(eval=merge_by_indices,
                                           doc=self.__doc__,
                                           name=name,
                                           parents=parents,
                                           #trace=False,
                                           #dtype=np.uint,
                                           *args, **kwds)


class KIndex(pymc.Deterministic):
    r""" A deterministic that dynamically tracks which time/first axis
    indices correspond to a given integer value.

    It keeps a static record of the indices it creates inside of
    any pymc.Node that is being indexed.  That means we don't create
    duplicates for the same k-index.

    XXX: Can't trace these Deterministics when they change shape.
    This might be a good reason for using a boolean mask instead.
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

        name = "which_{}_eq_{}".format(getattr(states, '__name__', 'states'),
                                       getattr(k, '__name__', k))

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
    """ Returns a tuple of indices and the partitions (of
    some array) corresponding to those indices.
    """
    res = None, None
    if isinstance(node, NumpyTake):
        idx = node.parents['indices']
        col = node.parents['a']
        res = (idx, col)
    elif isinstance(node, NumpyHstack):
        col = node.parents['tup']

        def ranges(parts):
            n = 0
            for p in parts:
                yield range(n, n + len(p))
                n += len(p)

        idx = list(ranges(col))
        res = (idx, col)
    elif isinstance(node, NumpyChoose):
        idx = node.parents['a']
        col = node.parents['choices']
        res = (idx, col)
    elif isinstance(node, HMMLinearCombination):
        idx = node.states
        col = node.k_prods
        res = (idx, col)
    elif isinstance(node, MergeIndexed):
        idx = node.parents['indices_']
        col = node.parents['values_']
        res = (idx, col)
    elif "index" in node.parents.keys():
        idx = node.parents['index']
        col = node.parents['self']
        res = (idx, col)

    return res
