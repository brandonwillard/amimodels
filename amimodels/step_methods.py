r"""
This module provides PyMC MCMC sampler step methods.

.. moduleauthor:: Brandon T. Willard
"""
import types
from warnings import warn
from itertools import chain

import numpy as np
import pymc
from pymc.LazyFunction import LazyFunction

import scipy.stats

from .stochastics import HMMStateSeq, TransProbMatrix
from .hmm_utils import compute_trans_freqs, compute_steady_state
from .deterministics import HMMLinearCombination


def create_lazy_logp_t(stoch):
    """ Create an element-wise logp function for a stochastic.

    Parameters
    ==========
    stoch: a pymc.Stochastic
        The stochastic variable from which to extract and create
        an element-wise logp function.

    Returns
    =======
        A LazyFunction for element-wise logp calculations.
    """

    if isinstance(stoch, pymc.Normal):

        def new_logp(*args, **kwargs):
            values = np.squeeze(kwargs.pop('value'))
            mu = np.squeeze(kwargs.pop('mu'))
            tau = np.squeeze(kwargs.pop('tau'))
            res = scipy.stats.norm.logpdf(values,
                                          loc=mu,
                                          scale=1/np.sqrt(tau))
            return res

    elif isinstance(stoch, pymc.TruncatedNormal):

        def new_logp(*args, **kwargs):
            values = np.squeeze(kwargs.pop('value'))
            a = np.squeeze(kwargs.pop('a'))
            b = np.squeeze(kwargs.pop('b'))
            mu = np.squeeze(kwargs.pop('mu'))
            tau = np.squeeze(kwargs.pop('tau'))
            res = scipy.stats.truncnorm.logpdf(values,
                                               a, b,
                                               loc=mu,
                                               scale=1/np.sqrt(tau))
            return res

    elif isinstance(stoch, pymc.HalfNormal):

        def new_logp(*args, **kwargs):
            values = np.squeeze(kwargs.pop('value'))
            tau = np.squeeze(kwargs.pop('tau'))
            res = scipy.stats.halfnorm.logpdf(values,
                                              scale=1/np.sqrt(tau))
            return res

    elif isinstance(stoch, pymc.Poisson):

        def new_logp(*args, **kwargs):
            values = np.squeeze(kwargs.pop('value'))
            mu = np.squeeze(kwargs.pop('mu'))
            res = scipy.stats.poisson.logpmf(values, mu)
            return res

    else:
        #raw_logp = stoch._logp_fun.func_closure[0].cell_contents
        raw_logp = stoch.raw_fns['logp']

        def single_or_indx(v, t, N):
            if N > 1 and np.alen(getattr(v, 'value', v)) == N:
                return v[t]
            else:
                return v

        def new_logp(*args, **kwargs):
            values = kwargs.pop('value')
            N_obs = np.alen(values)

            #res = np.array([raw_logp(y, **{k: single_or_indx(v, t, N_obs) for
            #                               k, v in kwargs.items()}) for t, y in
            #                enumerate(values)])
            res_iter = (raw_logp(y, **{k: single_or_indx(v, t, N_obs) for
                                       k, v in kwargs.items()}) for t, y in
                        enumerate(values))
            res = np.fromiter(res_iter, dtype=np.float, count=N_obs)

            return res

    # TODO: Could we apply something like the above to produce the following
    # "symbolic" dictionary?
    arguments = {}
    arguments.update(stoch.parents)
    arguments['value'] = stoch

    arguments = pymc.DictContainer(arguments)
    y_rv_logp_t = LazyFunction(fun=new_logp, arguments=arguments,
                               ultimate_args=stoch.extended_parents,
                               cache_depth=stoch._cache_depth)

    return y_rv_logp_t


def find_all_paths(start, end, path=[], ignored_nodes=[], dir='children'):
    """ Finds graph paths.

    Taken from `here <https://www.python.org/doc/essays/graphs/>`_.

    Parameters
    ==========
    start: a pymc.Stochastic
        Starting node.
    end: a pymc.Stochastic
        Terminating node.
    path: list
        Current path.
    dir: "children" or "parents"
        Direction of travel through the graph.

    Returns
    =======
        A list containing the discovered paths.

    """
    path = path + [start]
    if start == end:
        return [path]
    #if len(getattr(start, dir)) == 0:
    #    return []
    paths = []
    for node in getattr(start, dir):
        if node in ignored_nodes:
            continue
        if node not in path:
            newpaths = find_all_paths(node, end,
                                      path, ignored_nodes)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


class PriorObsSampler(pymc.StepMethod):
    """ This class will hack an observed variable with missing values
    so that it samples the entire variable (and not just the missing
    parts) from its prior.
    """

    def __init__(self, variables, verbose=-1, tally=True):
        super(PriorObsSampler, self).__init__(variables, verbose, tally=tally)

        #if not all([s_.mask is not None for s_ in self.stochastics]):
        #    raise ValueError(("At least one stochastic is not observed and/or"
        #                      " without a mask"))

        def get_stoch_value(self):
            return self._value

        for s_ in self.stochastics:
            s_.get_stoch_value = types.MethodType(get_stoch_value, s_)

    def step(self):
        for s_ in self.stochastics:
            s_.random()


class PosteriorSampler(pymc.StepMethod):
    """ If a stochastic has a non-empty `posterior` field,
    sample it!
    """
    @classmethod
    def competence(cls, targets):
        if all([isinstance(getattr(t_, 'posterior', None), pymc.Stochastic)
                for t_ in targets]):
            return 4
        else:
            return 0

    def __init__(self, variables, verbose=-1, tally=True):
        super(PosteriorSampler, self).__init__(variables, verbose, tally=tally)

    def step(self):
        for s_ in self.stochastics:
            s_.value = s_.posterior.value



class ExtStepMethod(pymc.StepMethod):
    r""" A pymc.StepMethod subclass that performs basic input consistency checks.
    Members
    =======
    target_classes: collection of class objects
        The stochastics class types that this step method
        can produce samples.
    child_classes: collection of class objects
        The stochastic class types that are dependent on the
        `target_classes` and can be used by this step method to
        produce informed samples.
    linear_OK: bool
        Not really used.
    target_exclusive_match: bool
        When determining the competence of a target stochastic,
        must we strictly match the `target_classes` in order
        to give non-zero competence?
    child_exclusive_match: bool
        When determining the competence of a target stochastic,
        must we strictly match the `child_classes` in order
        to give non-zero competence?
    children_conditioned: list of pymc.Node
        Nodes that this step method considers constants/conditionally.
    """

    child_classes = None
    target_classes = None
    linear_OK = False
    target_exclusive_match = False
    child_exclusive_match = False
    children_conditioned = []

    @staticmethod
    def valid_stochastic(stochastics, children):
        # TODO: Move any validation logic in the class constructors
        # to this location.
        return True

    @classmethod
    def competence(cls, targets):
        """ This class method checks that target and child classes
        match for the given step function.
        We have to do some work to get the (potential) child classes, though.
        """

        if not np.iterable(targets) or isinstance(targets, pymc.Node):
            targets = [targets]

        res = 4

        #valid_targets = (t_ for t_ in targets
        #                 if isinstance(t_, cls.target_classes))
        valid_targets = [isinstance(t_, cls.target_classes) and
                         not hasattr(t_, 'posterior')
                         for t_ in targets]

        if cls.target_exclusive_match and not any(valid_targets):
            return 0

        # No children?
        if len(cls.child_classes) == 0:
            warn("{} specifies no child classes".format(cls.__name__))
            return 0

        # We weren't given the children (that's usually done
        # in the class constructor), so we have to find them
        # to perform this check.
        children = chain(*(t_.extended_children | t_.children
                           for t_ in targets))

        #valid_children = (c_ for c_ in children
        #                  if isinstance(c_, cls.child_classes))
        valid_children = [isinstance(c_, cls.child_classes) for c_ in children]

        # TODO: Might want to relax these observed children
        # constraints.
        if len(valid_children) == 0:
            return 0

        if not all(valid_children):
            if cls.child_exclusive_match:
                return 0

            # This is a tough one: should we really take a point
            # off for not matching *all* child types?
            # In some cases the non-matching could be unobserved,
            # while the matching are observed.  That could make perfect
            # sense, and shouldn't lower the competence.
            if any(valid_children):
                #obs_valid_children = (getattr(c_, 'observed', False)
                #                      for c_ in valid_children)
                #if not all(obs_valid_children):
                #    res -= 1
                pass
            else:
                return 0

        return max(res, 0)

    def __init__(self, variables, *args, **kwargs):

        super(ExtStepMethod, self).__init__(variables, *args, **kwargs)

        # If we didn't exclusively match on child classes,
        # then we at least need to get rid of the ones we won't use.
        if self.child_classes is not None:
            children_active = set([c_ for c_ in self.children
                                   if isinstance(c_, self.child_classes)])
            self.children_conditioned = self.children - children_active
            self.children = children_active


class HMMStatesStep(ExtStepMethod):
    r"""
    Step method for HMMStateSeq stochastics.

    The model is given by

    .. math::

        \xi_k &\sim \operatorname{Dir}(\alpha_k), \quad k \in \{1,\dots,K\}\\
        S_{i,k} &\sim \operatorname{Cat}(\xi_k)


    This step method samples

    .. math::

        S_T &\sim p(S_T | y_{1:T}) \\
            S_t \mid S_{t+1}  &\sim p(S_{t+1} | S_t) p(S_{t+1} \mid y_{1:T})


    It's basically a forward-backward sampler.

    This step method is also designed to handle missing values (via masked
    observations) and piecewise state sequences (i.e. sequences of varying
    length that match tail-to-head).

    .. [fs] Sylvia Fruehwirth-Schnatter, "Markov chain Monte Carlo estimation of classical and dynamic switching and mixture models", Journal of the American Statistical Association 96 (2001): 194--209

    """
    child_classes = (pymc.Node,)
    target_classes = (HMMStateSeq,)
    linear_OK = False
    target_exclusive_match = True

    def __init__(self, variables, *args, **kwargs):
        """
        Parameters
        ==========
        variables: list of HMMStateSeq
            The order of the list determines the order of the
            entire (combined) sequence.
        """
        super(HMMStatesStep, self).__init__(variables, *args, **kwargs)

        self.state_seq = self.stochastics

        #N_obs = stoch.parents['N_obs']
        #N_obs = getattr(N_obs, 'value', N_obs)
        self.N_obs = sum([np.alen(v_.value) for v_ in self.state_seq])

        from collections import defaultdict
        self.stoch_to_obsfn = defaultdict(list)
        for stoch in self.state_seq:
            obs_children = self.children.intersection(stoch.extended_children)
            # This is fairly hackish, but we order multiple observed
            # stochastics by their names:
            for obs_var in sorted(obs_children, key=lambda x: x.__name__):
                self.stoch_to_obsfn[stoch].append((obs_var,
                                                   create_lazy_logp_t(obs_var))
                                                  )

        # Any connecting terms between obs_vars and our self.stochastic
        # (mus, for instance) should be immediately affected by changes to
        # self.stochastic.value, so we just need to set self.stochastic.value
        # to each possible state and then use the automatically computed
        # derived values (e.g. mus[].value) of the connecting terms.

        self.Ks = np.unique([s_.K for s_ in self.state_seq])

        if len(self.Ks) > 1:
            raise ValueError(("Stochastics should have same number of"
                              " categories"))

        self.y_logp_vals = np.empty((max(self.Ks), self.N_obs), dtype=np.float)
        self.logp_filtered = np.empty((self.N_obs, max(self.Ks)),
                                      dtype=np.float)
        self.p_filtered = np.empty((self.N_obs, max(self.Ks)), dtype=np.float)

    def compute_y_logp(self):
        """ Set all state sequence stochastics that, when concatenated/indexed,
        comprise a list that is in 1:1 correspondence with the entire
        observation sequence.
        """
        t_last = 0
        for stoch in self.state_seq:
            time_range = xrange(t_last, t_last + np.alen(stoch.value))

            obs_vars_to_funcs = self.stoch_to_obsfn[stoch]

            # TODO: How do we handle N_obs-many separate obs_stoch?
            # Right now we're assuming each obs_stoch has shape
            # (N_obs,) or sum to that shape.  This won't work
            # for split-up bivariate observations.
            if len(obs_vars_to_funcs) == self.N_obs or\
                    len(obs_vars_to_funcs) == np.alen(stoch.value):
                obs_vars_to_funcs = obs_vars_to_funcs[time_range]

            sum_mask = None
            logp_vals_sum = np.zeros_like(stoch.value, dtype=np.float)
            for k in xrange(self.Ks):
                for i, (obs_stoch, logp_fn) in enumerate(obs_vars_to_funcs):

                    mask = getattr(obs_stoch, 'mask', None)
                    if mask is not None:
                        mask = mask.ravel()
                        if sum_mask is None:
                            sum_mask = mask
                        else:
                            sum_mask |= mask

                    stoch.value = np.tile(k, len(time_range))

                    logp_vals = logp_fn.get()
                    if mask is not None:
                        logp_vals[mask] = 0.

                    #
                    # Take the product of each array of likelihoods,
                    # taking consideration of missing data/masked values.
                    #
                    if np.alen(logp_vals) == np.alen(stoch.value):
                        logp_vals_sum += logp_vals
                    else:
                        logp_vals_sum[i] += logp_vals

                    self.y_logp_vals[k][time_range] = logp_vals_sum

                logp_vals_sum[:] = 0.

            t_last += np.alen(stoch.value)

    def step(self):

        self.compute_y_logp()

        t_last = 0
        for stoch in self.state_seq:
            t_end = t_last + np.alen(stoch.value)
            time_range = xrange(t_last, t_end)

            trans_mat = stoch.parents['trans_mat']
            trans_mat = getattr(trans_mat, 'value', trans_mat)

            P = np.column_stack((trans_mat, 1. - trans_mat.sum(axis=1)))

            p0 = stoch.parents['p0']
            p0 = getattr(p0, 'value', p0)
            if p0 is None:
                p0 = compute_steady_state(trans_mat)

            p_run = p0
            # Very inefficient forward pass:
            for t in time_range:
                logp_k_t = self.logp_filtered[t]
                for k in xrange(stoch.K):
                    # This is the forward step (in log scale):
                    # p(S_t=k \mid y_{1:t}) \propto p(y_t \mid S_t=k) *
                    #   p(S_t=k \mid y_{1:t-1})
                    logp_k_t[k] = self.y_logp_vals[k, t] +\
                        pymc.categorical_like(k, p_run)

                # Here we normalize across k
                logp_k_t -= reduce(np.logaddexp, logp_k_t)

                # This computes p(S_{t+1} \mid y_{1:t})
                p_run = np.dot(np.exp(logp_k_t), P)

            np.exp(self.logp_filtered, out=self.p_filtered)

            # An inefficient backward pass:
            # Sample p(S_T \mid y_{1:T})
            new_values = np.empty_like(stoch.value, dtype=stoch.value.dtype)
            new_values[t_end-1] = pymc.rcategorical(self.p_filtered[t_end-1][:-1])
            for t in xrange(t_end-2, t_last-1, -1):
                # Now, sample p(S_t \mid S_{t+1}, y_{1:T}) via the relation
                # p(S_t=j \mid S_{t+1}=k, y_{1:T}) \propto
                #   p(S_t=j \mid S_{t_1}=k, y_{1:t}) \propto
                #   p(S_{t+1}=k \mid S_t=j, y_{1:t}) * p(S_t=j \mid y_{1:t})
                p_back = P[:, int(new_values[t + 1])] * self.p_filtered[t]
                p_back /= p_back.sum()

                new_values[t-t_last] = pymc.rcategorical(p_back[:-1])

            stoch.value = new_values

            t_last += np.alen(stoch.value)


class TransProbMatStep(ExtStepMethod):
    r'''
    Step method for a transition probability matrix (i.e. Dirichlet Prior rows
    each corresponding to [conditionally] Categorical observed variables).

    The model is given by

    .. math::

        \xi_k &\sim \operatorname{Dir}(\alpha_k), \quad k \in [1,\dots,K]\\
        S_{i,k} &\sim \operatorname{Cat}(\xi_k)


    This step method samples

    .. math::

        \xi^{(i)}_j &\sim p(\xi_j \mid S^{(i)}, y_{1:T}) \\
        &\sim \operatorname{Dir}(\left\{ \alpha_{j,k} + N_{j,k}(S) \right\}_{k=1}^K)


    The :math:`N_{j,k}(S)` are counts of observed transitions :math:`j \to k`.

    '''
    child_classes = (HMMStateSeq,)
    target_classes = (TransProbMatrix,)
    linear_OK = False
    target_exclusive_match = True

    def __init__(self, variables, *args, **kwargs):
        super(TransProbMatStep, self).__init__(variables, *args, **kwargs)

        if len(self.stochastics) != 1:
            raise NotImplementedError(("Implementation doesn't handle multiple"
                                       " stochastics"))

        (self.stochastic,) = self.stochastics

    def step(self):

        def masked_values(stoch):
            mask = getattr(stoch, 'mask', None)
            if mask is not None:
                return stoch.value[mask]
            else:
                return stoch.value

        S_cond = np.concatenate([masked_values(c_) for c_ in self.children])
        N_mat = compute_trans_freqs(S_cond, self.stochastic.shape[0],
                                    counts_only=True)

        # We assume, of course, that the parent(s) are the
        # alpha Dirichlet concentration parameters.
        alpha_suff_prior = self.stochastic.parents.values()[0]
        alpha_suff_prior = getattr(alpha_suff_prior, 'value', alpha_suff_prior)

        alpha_suff_post = alpha_suff_prior + N_mat

        trans_mat_smpl = np.zeros(self.stochastic.shape)
        for s, alpha_row in enumerate(alpha_suff_post):
            trans_mat_smpl[s] = pymc.rdirichlet(alpha_row)

        self.stochastic.value = trans_mat_smpl


class NormalNormalStep(ExtStepMethod):
    r'''
    Step method for Normal Prior with Normal likelihood.

    .. math::

        \beta &\sim \operatorname{N}(b, 1/\tau_b)\\
        y &\sim \operatorname{N}(X \beta, 1/\tau_y)
        \;.

    This step method samples exactly from the posterior
    :math:`p(\beta \mid y) \sim \operatorname{N}(m, 1/C)` using
    the conjugate Bayes--or equivalently the Kalman--update.

    '''
    child_classes = (pymc.Normal, pymc.TruncatedNormal)
    target_classes = (pymc.Normal, pymc.HalfNormal, pymc.TruncatedNormal)
    linear_OK = True
    target_exclusive_match = True

    def __init__(self, variables, *args, **kwargs):

        super(NormalNormalStep, self).__init__(variables, *args, **kwargs)

        try:
            (self.stochastic,) = self.stochastics
        except ValueError as e:
            raise NotImplementedError(("Step method only valid for a single "
                                       "stochastics:{}".format(e)))
        try:
            (self.obs_rv,) = filter(lambda x_: getattr(x_, 'observed', False),
                                    self.children)
        except ValueError as e:
            raise NotImplementedError(("Step method only valid for a single "
                                       "observed node."
                                       ":{}".format(e)))

        # If we can work with a marginal of our observed variable (that is
        # also the exclusive connection between stochastic and observed),
        # then do it!
        if hasattr(self.obs_rv, 'partitions'):
            parts_obs_rvs = filter(lambda x_:
                                   self.stochastic in x_.extended_parents,
                                   self.obs_rv.partitions)
            if len(parts_obs_rvs) == 1:
                self.obs_rv, = parts_obs_rvs

        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

        try:
            (beta_to_y_path,) = find_all_paths(self.stochastic, self.obs_rv,
                                               ignored_nodes=self.children_conditioned)
        except ValueError as e:
            raise NotImplementedError(("Step method only valid for a single "
                                       "linear relation to its observed child"
                                       ":{}".format(e)))

        # If it has no 'mu' parameter, then it's probably
        # the half-normal (with mean 0).
        self.mu_y = self.obs_rv.parents.get('mu', 0.)
        self.mu_beta = beta_to_y_path[1:-1]

        # XXX, TODO: mu_y and mu_beta might not be the same!
        # If they're both linear transformations, that
        # should be OK, though.  Just need to compose
        # all the transformations between them.

        assert len(self.mu_beta) == 2 and self.mu_y == self.mu_beta[-1]

        (self.X,) = self.mu_beta[0].x
        self.tau_y = self.y_beta.parents['tau']

        #
        # Check for a mask corresponding to missing values.
        #
        y_mask = getattr(self.y_beta, 'mask', None)
        if y_mask is not None and y_mask.any():
            y_mask = ~y_mask.squeeze()
        else:
            y_mask = None

        #
        # When our stochastic really applies to a subset of
        # observations, make sure we use only those.
        #
        if isinstance(self.mu_y, HMMLinearCombination):
            self.which_k = self.mu_y.beta_obs_idx[self.stochastic]

            if y_mask is not None:
                # self.X is already the subset of observations in this step's
                # state, so we just need to get rid of the missing values from
                # that subset.
                self.X = self.X[self.which_k[y_mask]]
                y_mask = self.which_k & y_mask
            else:
                y_mask = self.which_k

        if y_mask is not None:
            self.y_obs = self.y_beta[y_mask]

            if np.ndim(self.tau_y) == 2:
                self.tau_y = self.tau_y[np.ix_[y_mask, y_mask]]
            elif np.alen(getattr(self.tau_y, 'value', self.tau_y)) > 1:
                self.tau_y = self.tau_y[y_mask]

        # Again, if it has no 'mu' parameter, then it's probably
        # the half-normal (with mean 0).
        self.a_beta = self.stochastic.parents.get('mu', 0.)
        self.tau_beta = self.stochastic.parents['tau']

        self.post_a = None
        self.post_b = None

        if isinstance(self.y_beta, (pymc.HalfNormal, pymc.TruncatedNormal)):
            self.post_a = self.y_beta.parents.get('a', 0)
            self.post_b = self.y_beta.parents.get('b', np.inf)

        if isinstance(self.stochastic,
                      (pymc.HalfNormal, pymc.TruncatedNormal)):
            beta_a = self.stochastic.parents.get('a', 0)
            beta_b = self.stochastic.parents.get('b', np.inf)
            self.post_a = beta_a if self.post_a is None else\
                np.maximum(self.post_a, beta_a)
            self.post_b = beta_b if self.post_b is None else\
                np.minimum(self.post_b, beta_b)

        # Now, let's see if we can hack up a means of
        # finding the observation variable's precision
        # stochastic (if any).
        if isinstance(self.tau_y, pymc.Node):
            if isinstance(self.tau_y, pymc.Gamma):
                self.V_inv = self.tau_y
            else:
                # XXX: Very narrow approach.
                if set(self.tau_y.parents.keys()) == set(['index', 'self']):
                    k_ = self.tau_y.parents['index'].parents['k_']
                    self.V_inv = self.tau_y.parents['self'].parents['V_invs_'][k_]

    def step(self):
        # We're going to do this in a way that allows easy extension
        # to multivariate beta (and even y with non-diagonal covariances,
        # for whatever that's worth).

        y = np.atleast_1d(np.squeeze(self.y_obs.value))

        if np.alen(y) == 0:
            self.stochastic.random()
            return

        X = getattr(self.X, 'value', self.X)
        # Gotta broadcast when the parameters are scalars.
        bcast_beta = np.ones_like(self.stochastic.value)
        a_beta = bcast_beta * getattr(self.a_beta, 'value', self.a_beta)
        tau_beta = bcast_beta * np.atleast_1d(getattr(self.tau_beta, 'value',
                                                      self.tau_beta))

        tau_y = getattr(self.tau_y, 'value', self.tau_y)

        #
        # This is how we get the posterior mean:
        # C^{-1} m = R^{-1} a + F V^{-1} y
        #
        rhs = np.dot(tau_beta, a_beta) + np.dot(X.T * tau_y, y)

        tau_post = np.diag(tau_beta) + np.dot(X.T * tau_y, X)

        a_post = np.linalg.solve(tau_post, rhs)
        tau_post = np.diag(tau_post)

        # TODO: These could be symbolic/Deterministic, no?
        parents_post = {'mu': a_post, 'tau': tau_post}
        self.stochastic.parents_post = parents_post

        # TODO: If self.V_inv, sample normal-gamma dist

        if self.post_a is not None and self.post_b is not None:
            parents_post['a'] = self.post_a
            parents_post['b'] = self.post_b
            res = pymc.rtruncated_normal(**parents_post)

            # pymc's truncated distribution(s) doesn't handle
            # the limit values correctly, so we have to clip
            # the values.
            self.stochastic.value = res.clip(self.post_a, self.post_b)
        else:
            self.stochastic.value = pymc.rnormal(**parents_post)

        # TODO: We should consider setting the stochastic's parents.
        #
        # We would need to make sure those aren't deterministics with
        # further dependencies or stochastics with other assigned
        # step methods.
        #
        # Really, what underlies this are the much bigger ideas of having,
        # finding and using posteriors in a symbolic framework, and these don't
        # have well-defined places in the PyMC framework.


class GammaNormalStep(ExtStepMethod):
    r'''
    Step method for a Gamma Prior precision/scale and Normal likelihood.

    .. math::

        \tau_y &\sim \operatorname{Gamma}(n_0 / 2, n_0 S_0 / 2)\\
        y &\sim \operatorname{N}(\mu, 1/\tau_y)
        \;.

    This step method samples exactly from the posterior

    ..math::
        (\tau_y \mid y) \sim \operatorname{Gamma}(n_1/2, n_1 S_1/2)
        \text{where} \quad n_1 = n_0 + T, \; \text{and}\quad
        n_1 S_1 = n_0 S_0 + \sum_{t=0}^T (y_t - \mu_t)^2

    '''
    child_classes = (pymc.Normal, pymc.TruncatedNormal, pymc.HalfNormal)
    # TODO: add pymc.InverseGamma target
    target_classes = (pymc.Gamma, )
    linear_OK = False
    target_exclusive_match = True

    def __init__(self, variables, state_obs_mask=None, *args, **kwargs):
        r"""
        Parameters
        ----------
        variables: list of pymc.Stochastic
            Currently only handles a single pymc.Stochastic.
        """

        super(GammaNormalStep, self).__init__(variables, *args, **kwargs)

        try:
            self.stochastic, = self.stochastics
        except ValueError as e:
            raise NotImplementedError(("Step method only valid for a single "
                                       "stochastics:{}".format(e)))
        try:
            self.obs_rv, = filter(lambda x_: getattr(x_, 'observed', False),
                                  self.children)
        except ValueError as e:
            raise NotImplementedError(("Step method only valid for a single "
                                       "observed node."
                                       ":{}".format(e)))

        # If we can work with a marginal of our observed variable (that is
        # also the exclusive connection between stochastic and observed),
        # then do it!
        if hasattr(self.obs_rv, 'partitions'):
            parts_obs_rvs = filter(lambda x_:
                                   self.stochastic in x_.extended_parents,
                                   self.obs_rv.partitions)
            if len(parts_obs_rvs) == 1:
                self.obs_rv, = parts_obs_rvs

        # Let's check that our gamma stochastic is connected to its normal
        # observation in a conjugate way.
        self.tau_obs = self.obs_rv.parents['tau']

        # If it has no 'mu' parameter, then it's probably
        # the half-normal (with mean 0).
        mu_obs = self.obs_rv.parents.get('mu', 0.)

        # TODO: This should be in the competence check.
        if self.stochastic not in self.tau_obs.extended_parents or\
                self.stochastic in getattr(mu_obs, 'extended_parents', ()):
            raise NotImplementedError("Invalid child dependency")

        # TODO: If self.stochastic is 1-d (or a vector with self.obs_rv's
        # dimension) and only ever multiplied, then we can attempt to pull it
        # out of the collapsed product.
        self.obs_tau = 1
        if self.tau_obs is not self.stochastic:
            if '_mul_' in self.tau_obs.__name__ and\
                    self.stochastic in self.tau_obs.parents.values():
                self.obs_tau = filter(lambda x_: x_ is not self.stochastic,
                                      self.tau_obs.parents.values())
            else:
                raise NotImplementedError("Invalid child dependency")

        obs_idx = getattr(self.stochastic, 'obs_idx', None)

        # Check for a mask corresponding to missing values.
        missing_obs_mask = getattr(self.obs_rv, 'mask', None)
        if missing_obs_mask is not None and missing_obs_mask.any():
            missing_obs_idx = np.flatnonzero(~missing_obs_mask.squeeze())

            if obs_idx is not None:
                obs_idx = np.intersect1d(obs_idx, missing_obs_idx,
                                         assume_unique=True)
            else:
                obs_idx = missing_obs_idx

        if obs_idx is not None:
            self.gamma_obs = self.obs_rv[obs_idx]
            self.gamma_mu = mu_obs[obs_idx]
        else:
            self.gamma_obs = self.obs_rv
            self.gamma_mu = mu_obs

        self.alpha_prior = self.stochastic.parents['alpha']
        self.beta_prior = self.stochastic.parents['beta']

    def step(self):
        # We're going to do this in a way that allows easy extension
        # to multivariate beta (and even obs with non-diagonal covariances,
        # for whatever that's worth).

        y = np.atleast_1d(np.squeeze(self.gamma_obs.value))

        if np.alen(y) == 0:
            self.stochastic.random()
            return

        mu_y = getattr(self.gamma_mu, 'value', self.gamma_mu)

        r1 = y - mu_y
        r2 = np.dot((r1 * self.obs_tau), r1)

        alpha_post = self.alpha_prior + np.alen(y)/2.
        beta_post = self.beta_prior + r2/2.

        parents_post = {'alpha': alpha_post, 'beta': beta_post}

        self.stochastic.parents_post = parents_post

        self.stochastic.value = pymc.rgamma(**parents_post)
