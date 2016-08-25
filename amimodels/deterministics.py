r"""
This module provides PyMC Deterministic objects custom built
for hidden Markov models.

.. moduleauthor:: Brandon T. Willard
"""
import numpy as np
import pymc
import collections


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
        betas: list of np.ndarrays
            Vectors corresponding to each :math:`beta^{(S_t)}` in the
            product :math:`x_t^\top \beta^{(S_t)}`.
        states: np.ndarray of int
            Vector of the state sequence :math:`S_t`.
        """

        self.N_states = np.alen(X_matrices)

        if not self.N_states == np.alen(betas):
            raise ValueError("len(X_matrices) must equal len(betas)")

        if not np.count_nonzero(np.diff([np.alen(X_) for X_ in X_matrices])) == 0:
            raise ValueError("X_matrices must all have equal first dimension")

        self.N_obs = np.alen(X_matrices[0])

        if not self.N_obs == np.alen(states.value):
            raise ValueError("states do not match X_matrices dimensions")

        if not all([np.shape(X_)[-1] == np.alen(b_)
                    for X_, b_ in zip(X_matrices, betas)]):
            raise ValueError("X_matrices and betas dimensions don't match")

        self.which_k = tuple()
        self.X_k_matrices = tuple()
        self.k_prods = tuple()
        self.beta_obs_idx = dict()

        for k in xrange(self.N_states):

            def which_k_func(s_=states, k_=k):
                return np.equal(k_, s_)

            w_k = pymc.Lambda("w_{}".format(k), which_k_func)

            def X_k_func(X_=X_matrices[k], t_=w_k):
                return X_[t_]

            X_k_mat = pymc.Lambda("X_{}".format(k), X_k_func)

            this_beta = betas[k]
            k_p = pymc.LinearCombination("mu_{}".format(k),
                                         (X_k_mat,),
                                         (this_beta,))

            self.which_k += (w_k,)
            self.X_k_matrices += (X_k_mat,)
            self.k_prods += (k_p,)

            if isinstance(this_beta, collections.Hashable):
                self.beta_obs_idx[this_beta] = w_k

        def eval_fun(which_k, k_prods):
            res = np.empty(self.N_obs, dtype=np.float)
            for idx, k_p in zip(which_k, k_prods):
                res[idx] = k_p
            return res

        super(HMMLinearCombination, self).__init__(eval=eval_fun,
                                                   doc="",
                                                   name=name,
                                                   parents={'which_k': self.which_k,
                                                            'k_prods': self.k_prods},
                                                   *args, **kwds)
