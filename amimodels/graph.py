r"""
This module provides graph parsing methods for PyMC2 models.

.. moduleauthor:: Brandon T. Willard
"""
from functools import partial
from operator import (mul, pow, is_, itemgetter, isNumberType)
from itertools import (chain, cycle, izip)

import numpy as np
import pymc

from .stochastics import dvalue_class

from .deterministics import (NumpySum, NumpyBroadcastTo, NumpyAlen,
                             MergeIndexed, get_indexed_items)


class CallableNode(object):
    """ Simple class for delayed construction of a pymc.Distribution.

    XXX: This isn't a great way to go about it.  We probably just want
    nodes that don't necessarily add themselves as children to their
    parents (well, that *definitely* is something we want, among potentially
    other things).  Another way to go about this could involve a pymc.Node
    class that "detaches" itself from its parents immediately after creation.
    However, that doesn't stop us from all the other setup (e.g. construction
    of lazy log probability functions, etc.)

    We could also consider exclusively using unevaluated nodes of our own, but
    then we would end up re-writing the PyMC symbolic logic.  Might be worth it
    if we can use SymPy objects instead, then convert those to PyMC after all
    of this graph work.  On a similar note, PyMC3/Theano should be a lot better
    for this situation.
    """
    def __init__(self, dist_cls, parents, kwargs, name="", properties={}):
        self.dist_cls = dist_cls
        self.properties = properties
        self.parents = parents
        self.name = name
        self.kwargs = kwargs

        self._dvalue = self.kwargs.pop('value', None)
        if isinstance(self._dvalue, CallableNode):
            # If the value field is another CallableNode, then we need to
            # use its value object.  Otherwise, we're hijacking the same
            # "private" member that the dvalue Stochastic wrapper class uses.
            self._dvalue = getattr(self._dvalue, '_dvalue', self._dvalue)

        # XXX: This is another hack to emulate pymc.Node when we need the
        # information from this unevaluated Node.
        self.__dict__.update(kwargs)

    def __call__(self):
        eff_cls = self.dist_cls

        dist_kwargs = dict(self.parents.items() + self.kwargs.items())

        if dist_kwargs.get('observed', False):
            res = dvalue_class(eff_cls, self.name,
                               value=self._dvalue, **dist_kwargs)
        else:
            res = eff_cls(self.name, **dist_kwargs)

        res.__dict__.update(self.properties)

        return res

    @property
    def value(self):
        """ XXX: This is a hack that allows these objects to used like unevaluated
        pymc.Nodes.  It's not a great idea.
        """
        return getattr(self._dvalue, 'value', self._dvalue)


def izip_repeat(*args):
    arg_groups = [(len(v), list(v)) for v in args]
    largest_len = max(v[0] for v in arg_groups)
    adj_args = chain(iter(a) if l == largest_len else cycle(a)
                     for l, a in arg_groups)
    res = izip(*adj_args)
    return res


def node_eq(x, y):
    """ Compare nodes by parent structure and values, i.e.
    in a symbolic fashion.

        mu = pymc.Normal('mu_y', 0, 1, size=10)
        node = pymc.Normal('y', mu, 1, size=10)
        assert node_eq(node, node)
        assert not node_eq(node[0], node)
        assert node_eq(node[0], node[0])

        node_2 = pymc.Normal('y_2', mu, 1, size=10)
        assert node_eq(node_2, node)
        assert node_eq(node, node_2)

        mu.parents['mu'] = 4
        assert node_eq(node, node_2)
        assert node_eq(node_2, node)

        node_3 = pymc.Normal('y_3', mu, 2, size=10)
        assert not node_eq(node_3, node)

        node_4 = pymc.Normal('y_4', mu, np.ones(10), size=10)
        assert node_eq(node_4, node)

        assert node_eq(node_4[0:5], node_4[0:5])

        assert not node_eq(node_4[0:5], node_4[0:6])
    """
    if x is y:
        return True
    elif y is None:
        return False

    if isinstance(x, pymc.Node) and isinstance(y, pymc.Node):

        if type(x) != type(y):
            return False

        return all(node_eq(v_, y.parents.get(k_))
                   for k_, v_ in x.parents.iteritems())

    elif isNumberType(x) and isNumberType(y):
        return np.array_equiv(x, y)
    else:
        return x == y


def get_non_node(x, y, node):
    """ From two of three nodes, choose the one that doesn't match
    the excluded "base" node and throw an exception if the two
    choices are unique (and neither is the base node).

    This, when used with `reduce`, produces a filter that can be used
    to...

    Examples speak louder than words:

        from functools import partial
        mu = pymc.Normal('mu', 0, 1, size=10)
        node = pymc.Normal('y', mu, 1, size=10)
        cmp_nodes = partial(get_non_node, node=node)

        # These are ok
        reduce(cmp_nodes, [node, node[range(1)], node, node])
        # <pymc.PyMCObjects.Deterministic 'y[[0]]' at 0x7fd608a5f890>

        reduce(cmp_nodes, [node[range(1)]])
        # <pymc.PyMCObjects.Deterministic 'y[[0]]' at 0x7fd608a5f890>

        reduce(cmp_nodes, [node, node])
        # <pymc.distributions.Normal 'y' at 0x7fd5e4a27210>

        reduce(cmp_nodes, [node])
        # <pymc.distributions.Normal 'y' at 0x7fd5e4a27210>

        reduce(cmp_nodes, [node[range(1)], node[range(1)]])
        # <pymc.PyMCObjects.Deterministic 'y[[0]]' at 0x7fd608a5f890>

        # XXX: Not OK!
        reduce(cmp_nodes, [node[range(1)], node[range(3)]])

    """
    if isinstance(x, pymc.Node) and\
            isinstance(y, pymc.Node):
        if node_eq(x, y):
            return x

        if not node_eq(x, node):
            if not node_eq(y, node):
                raise ValueError('Heterogeneous nodes')
            return x
        else:
            return y
    else:
        if np.alen(x) < np.alen(y):
            return x
        else:
            return y


def static_vars(**kwargs):
    """ Decorator for "function-static" variables.

    Might be worth trying with our `put`/array setting
    deterministics; perhaps we could save on allocations.
    XXX: Would very likely ruin any asynchronous use of such
    deterministics, though.

    .. _source:
        http://stackoverflow.com/a/279586

    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_mul_type(node, allow_pow=False):
    """
        node = pymc.Normal('', 0, 1, size=4)
        n = node * node
        get_mul_type(n)
        # <ufunc 'multiply'>
    """
    if isinstance(node, pymc.Deterministic):
        if isinstance(node, pymc.LinearCombination):
            return np.dot
        try:
            func = node._value.fun.__closure__[0].cell_contents
            if func is mul:
                return np.multiply
            elif allow_pow and func is pow:
                return np.power
        except:
            pass

    return None


def extract_term_mul(node, func=lambda *args: False,
                     m_type_last=None):
    """ Creates a list of nodes used in a--potentially nested--
    deterministic product.
    Using ``func`` is like dividing by whatever yields func == True.

    For example:

        extract_term_mul(1)
        # (1, None)

        extract_term_mul(np.ones(2))
        # (array([ 1.,  1.]), None)

        node_1 = pymc.Normal('y', 0, 1, size=10)
        extract_term_mul(node_1)
        # [(<pymc.distributions.Normal 'y' at 0x7f87ad78d150>, <ufunc 'multiply'>),
        # (2, None)]

        extract_term_mul(node_1 * 2)
        # [(<pymc.distributions.Normal 'y' at 0x7f87ad78d150>, <ufunc 'multiply'>),
        # (2, None)]

        assert set(extract_term_mul(node_1 * 2 * 5)) ==\
            set(extract_term_mul(2 * node_1 * 5))

        extract_term_mul(node_1 * 2 * 5)
        # [(<pymc.distributions.Normal 'y' at 0x7f87ad78d150>, <ufunc 'multiply'>),
        #  (2, <ufunc 'multiply'>),
        #  (5, None)]

        extract_term_mul(node_1 * 2 * 5, func=lambda x: x is node_1)
        #  [(None, <ufunc 'multiply'>), (2, <ufunc 'multiply'>), (5, None)]

        extract_term_mul(3 * node_1 * node_1[3] * 2,
                         func=lambda x: node_eq(x, node_1[3]))
        # [(3, <ufunc 'multiply'>),
        #  (<pymc.distributions.Normal 'y' at 0x7f87ad78d150>, <ufunc 'multiply'>),
        #  (None, <ufunc 'multiply'>),
        #  (2, None)]

        node_lc = pymc.LinearCombination('', (np.ones((2, 2)),), (node_1[0:2],))
        extract_term_mul(3 * node_lc * 2)
        # [(3, <ufunc 'multiply'>), (array([[ 1.,  1.],
        #          [ 1.,  1.]]),
        #   <function numpy.core.multiarray.dot>), (<pymc.PyMCObjects.Deterministic 'y[slice(0, 2, None)]' at 0x7f05e6eeb890>, <ufunc 'multiply'>), (2,
        #   None)]

        extract_term_mul(3 * node_lc * 2, func=lambda x: node_eq(x, node_1[0:2]))

        # [(3, <ufunc 'multiply'>), (array([[ 1.,  1.],
        #          [ 1.,  1.]]), <function numpy.core.multiarray.dot>), (None,
        #   <ufunc 'multiply'>), (2, None)]

        # Doesn't handle powers...
        assert set(extract_term_mul(node_1 * node_1)) !=\
            set(extract_term_mul(node_1**2))

        # It should also maintain the order of terms:
        from operator import mul
        extract_term_mul(reduce(mul, [node_1] + range(5)))
        # [(<pymc.distributions.Normal 'y' at 0x7f05e704c390>, <ufunc 'multiply'>),
        #  (0, <ufunc 'multiply'>),
        #  (1, <ufunc 'multiply'>),
        #  (2, <ufunc 'multiply'>),
        #  (3, <ufunc 'multiply'>),
        #  (4, None)]

        # TODO: Get NumpyBroadcastArrays and NumpyMultidot working.
    """
    m_type = get_mul_type(node)

    # pymc.LinearCombination fix:
    if isinstance(node, (list, tuple)):
        node, = node

    if m_type is not None:
        mul_args = sorted(node.parents.iteritems(),
                          key=itemgetter(0))
        res = (extract_term_mul(mul_args[0][1],
                                func, m_type_last=m_type),
               extract_term_mul(mul_args[1][1],
                                func, m_type_last=m_type_last))
        return list(chain.from_iterable(res))
    elif not func(node):
        return ((node, m_type_last),)
    else:
        return ((None, m_type_last),)


def node_shape(x):
    if isinstance(x, pymc.Deterministic):
        return np.shape(pymc.utils.value(x))
    else:
        return np.shape(x)


def commute(prod_elems, func):
    """ Commute a term in a list as far to the right as possible.

    Parameters
    ==========
    prod_elems: iterable of `(node, prod_type)`
        An iterable containing tuples of node and product-type elements.
        Product-type elements are one of (np.dot, np.multiply).
        Really whatever `extract_term_mul` returns.
    func: function
        A single argument boolean function that identifies the term to be
        commuted.

    Returns
    =======
    iterable of `(node, prod_type)`
        An iterable with the same elements as ``prod_elems`` but with
        the ``func`` identified term commuted, if possible.


    Examples
    ========

        node = pymc.Normal('', 0, 1, size=4)

        func = lambda x_: isinstance(x_, type(node))

        prod_elems = ((pymc.Lambda('', lambda x_=np.ones((5,4)): x_), np.dot),
                    (node, np.multiply),
                    (np.ones(5), None))

        commuted_elems = commute(prod_elems, func)

        def check_commute(start_elems, end_elems):
            start_res = reduce(lambda a, b: (a[1](pymc.utils.value(a[0]),
                                                pymc.utils.value(b[0])), b[1]),
                            start_elems)
            end_res = reduce(lambda a, b: (a[1](pymc.utils.value(a[0]),
                                                pymc.utils.value(b[0])), b[1]),
                            end_elems)

            return start_res[0], end_res[0]

        assert np.array_equal(*check_commute(prod_elems, commuted_elems))

        prod_elems = ((pymc.Lambda('', lambda x_=np.ones((4,4)): x_), np.dot),
                    (node, np.multiply),
                    (np.ones(4), None))

        commuted_elems = commute(prod_elems, func)
        assert np.array_equal(*check_commute(prod_elems, commuted_elems))

        prod_elems = ((pymc.Lambda('', lambda x_=np.ones((4,4)): x_), np.dot),
                    (node, np.multiply),
                    (np.array(1), None))

        commuted_elems = commute(prod_elems, func)
        assert np.array_equal(*check_commute(prod_elems, commuted_elems))

        prod_elems = ((np.array(1), None),)

        commuted_elems = commute(prod_elems, func)
        assert np.array_equal(*check_commute(prod_elems, commuted_elems))

        prod_elems = ((np.array(1), None),)

        commuted_elems = commute(prod_elems, func)
        assert np.array_equal(*check_commute(prod_elems, commuted_elems))

    """
    if len(prod_elems) == 1:
        return prod_elems
    # Track whether we've seen a dot product in the chain.
    # We need this to signal that we've lost shape information.
    # TODO: If the chain contains only arrays and stochastics, we
    # might actually be able to track shape information.
    seen_dot = False

    res = [prod_elems[0]]
    for b in prod_elems[1:]:
        a_ = res[-1]
        a_val = a_[0]
        a_mul_type = a_[1]

        if a_mul_type is np.dot:
            seen_dot = True

        b_val = b[0]
        b_mul_type = b[1]

        # We can't get reliable enough shape information for deterministics,
        # so we assume they can't commute.
        if func(a_val):
            a_shape = np.shape(a_val) if (isinstance(a_val, pymc.Stochastic) or
                                          not isinstance(a_val, pymc.Node)) else None

            b_shape = np.shape(b_val) if (isinstance(b_val, pymc.Stochastic) or
                                          not isinstance(b_val, pymc.Node)) else None

            # Scalars can commute with anything.
            if a_shape == () or b_shape == ():
                if len(res) > 1:
                    # Unless it's all scalar products, the product before these
                    # two need to be scalar multiplied, and the result of that
                    # dotted or whatever else.
                    ur_res = res[-2]
                    res[-2] = (ur_res[0], a_mul_type)
                    res[-1] = (b_val, ur_res[1])
                else:
                    res[-1] = (b_val, a_mul_type)
                res += [(a_val, b_mul_type)]

            # Otherwise, we can commute when Hadamard multiplying equal shapes.
            elif not seen_dot and (a_shape == b_shape and
                                   a_mul_type is np.multiply):
                res[-1] = (b_val, a_mul_type)
                res += [(a_val, b_mul_type)]
        else:
            res += [b]

    return res


def get_linear_parts(node, func, mu_name=''):
    """ Creates a pymc.LinearCombination from a
    nested product and attempts to make one of the terms
    the node that makes ``func`` return True.

    Parameters
    ==========
    node: pymc.Node
        Potential linear combination node.
    func: function
        Filter for determining right product term.
    Returns
    =======
    pymc.LinearCombination
    """
    prod_elems = extract_term_mul(node)

    if len(prod_elems) <= 1:
        left_term = 1
        right_term = node
    else:
        commuted_elems = commute(prod_elems, func)
        left_term = reduce(lambda a, b: (a[1](a[0], b[0]), b[1]),
                           commuted_elems[:-1])[0]
        right_term = commuted_elems[-1][0]

    mu_k = pymc.LinearCombination(mu_name, (left_term,),
                                  (right_term,))

    return mu_k


# TODO: Use something like LogPy.
conjugate_relations = {pymc.Normal: {'mu': pymc.Normal, 'tau': pymc.Gamma}}


def parse_partition(node, parent, parent_name):
    """ Creates the partitioned LinearCombinations of a node's parents.

    Parameters
    ==========
    node: pymc.Node
        The node.
    parent: pymc.Node
        Single parent of ``node``
    parent_name: str
        Name of parent node.

    Returns
    =======
    tuple
    """

    #
    # Check for a mask corresponding to missing values.
    #
    node_mask = getattr(node, 'mask', None)
    if node_mask is not None and np.any(node_mask):
        node_mask = np.squeeze(node_mask)
    else:
        node_mask = None

    #
    # When our stochastic really applies to a subset of
    # observations, make sure we use only those.
    #
    parts = ()
    if isinstance(parent, pymc.LinearCombination):

        X, = parent.x
        beta, = parent.y
        mu_name = '{}_{}'.format(parent_name, node.__name__)
        mu = pymc.LinearCombination(mu_name,
                                    [X], [beta])
        parts += ({parent_name: mu, 'value': node},)

    elif isinstance(parent, pymc.Deterministic):
        # Parse indexed collections of linear combinations
        # (or things that can be turned into linear combinations).

        k_idx_s, mu_s = get_indexed_items(parent)

        for k, (mu_k_, k_idx) in enumerate(zip(mu_s, k_idx_s)):

            mu_name = '{}_{}_{}'.format(parent_name, node.__name__, k)
            conj_dist = conjugate_relations.get(type(node), {})
            conj_dist = None if conj_dist is None else conj_dist.get(parent_name)
            mu_k = get_linear_parts(mu_k_,
                                    lambda x_: isinstance(x_, conj_dist),
                                    mu_name)

            if k_idx is not None:
                node_k = node[k_idx]
            else:
                node_k = node

            if node_mask is not None:
                if k_idx is not None:
                    y_k_mask = pymc.Lambda('{}_idx'.format(k),
                                           lambda k_m_=k_idx, y_m_=node_mask:
                                           y_m_[k_m_], trace=False)
                else:
                    y_k_mask = node_mask

                node_k._mask = y_k_mask

            parts += ({parent_name: mu_k, 'value': node_k},)

    else:
        mu_name = '{}_{}'.format(parent_name, node.__name__)
        mu = pymc.LinearCombination(mu_name,
                                    [1], [parent])
        parts += ({parent_name: mu, 'value': node},)

    return parts


def parse_node_partitions(node):
    """ Creates the collections of partitioned nodes, if any, needed to
    construct a marginal and/or posterior.

    This scheme accounts for discrete, stochastic partitions of
    observation indices, as seen in Hidden Markov Models, etc.

    This function only applies to univariate distributions
    and assumes that partitions on the first dimension of a node
    must also apply to the first dimension of its parents, and
    vice versa.

    .. todo:
        The last condition is probably too strong.

    It starts by trying to identify partitions in the parents of a
    source ``node``.  Those partitions are returned as a list of dicts
    like the following:

        ({'<parent_name>': parent_node_part_1, 'value': value_node_part_1},
         ...,
         {'<parent_name>': parent_node_part_n, 'value': value_node_part_n})

    We then take the list of these dicts, for each parent, and zip them
    together, naively repeating the shorter lists, so that we ultimately have
    a list of dicts with parent name and node pairs.  Since there are multiple
    'value' items, we have to choose one of them so that we have the
    kwargs necessary to produce distributions for each partition of ``node``.

    .. todo:
        We should actually "broadcast" the shorter lists; checking
        that the dimensions match.

    Simply put, we want the "finest" partitioning found among all the parent nodes.
    However, there can be multiple, non-identical "finest" partitions.

    Take a pymc.Normal with size > 1 and a 'mu' parameter node that's partitioned
    into N-many parts via indexing by some sequence/array along the first axis.
    If the 'tau' parameter node is similarly partitioned into N-many parts,
    but by a different sequence/array, we are left with multiple partitions
    for the base ``node``.

    Parameters
    ==========
    node: pymc.Node
        The node.

    Returns
    =======
    tuple
        Collection of partitions of ``node``.
    """

    parents_parts = [np.asarray(parse_partition(node, v_, k_))
                     for k_, v_ in node.parents.items()]

    def unique_pairwise(iterable):
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for e in it:
            total = total if node_eq(total, e) else e
            yield total

    parent_indices = [list(unique_pairwise(map(itemgetter('value'), p_p)))
                      for p_p in parents_parts]

    # FIXME: Only using the values from the first list `max` picks!
    value_parts = max(parent_indices, key=len)

    zipped_parts = np.broadcast_arrays(*parents_parts)
    zipped_parts = np.asarray(zipped_parts).T

    node.partitions = ()
    for n, (n_parents, n_value) in enumerate(zip(zipped_parts,
                                                 value_parts)):

        parents = dict(chain.from_iterable(
            [(k, v) for k, v in z.items() if k is not 'value']
            for z in n_parents))

        kwargs = dict()
        kwargs.update({'value': n_value,
                       'observed': node.observed})
        properties = {"_mask": getattr(n_value, "_mask", None)}

        part_name = "{}-{}".format(node.__name__, n)
        part_node = CallableNode(type(node), parents, kwargs,
                                 name=part_name,
                                 properties=properties)
        node.partitions += (part_node,)

    return node.partitions


def normal_node_update(node, updates):
    """ Constructs the marginal for a given node and
    the posteriors for some of its parents (when possible).

    Parameters
    =========
    node: pymc.Node
        The node for which we want to produce marginals.
    updates: collection
        This parameter is the output of `parse_normal` for ``node``.

    Returns
    =======
    None
        This function adds fields `*.marginals` and `*.posteriors` to ``node``
        and its applicable parents, repectively.
    """

    node_marginals = ()

    for k, node_k in enumerate(updates):
        mu_k = node_k.parents['mu']
        tau_k = node_k.parents['tau']

        # We can get an index from a node's value when that value is
        # "dynamic" (e.g. another symbolic node); otherwise, try to get it directly
        # from the node (again, assuming the node is an "indexing"
        # Deterministic).
        node_idx, _ = get_indexed_items(getattr(node_k,
                                                '_dvalue',
                                                node_k))

        # The right-most term should be a gamma, if any.
        if isinstance(tau_k, pymc.LinearCombination):
            tau_k_gam, = tau_k.y
        else:
            tau_k_gam = tau_k

        # Log the terms we are marginalizing.
        marginal_wrt = ()

        if isinstance(mu_k.y[0], pymc.Normal):
            # Normal-Normal Update
            beta_k, = mu_k.y
            X_k, = mu_k.x

            # Log the term we are marginalizing.
            marginal_wrt += (beta_k,)

            # Get a delayed/promised marginal distribution object.
            # If we create it now, then it will become a dependency/child
            # of all its parents, and that can be super annoying when
            # only the marginal moments are needed.
            node_marginal = norm_norm_marginal(beta_k, node_k, mu_k, tau_k)

            #
            # Compute the mu/mean/beta posterior.
            #
            # TODO: This should probably be its own thing (e.g. in a pass
            # through the entire graph we would do this when arriving at the
            # mu/mean/beta node), but we're doing it here because it's
            # convenient.
            beta_post = norm_norm_post(beta_k, node_k, mu_k, tau_k)

            # (Re)set these so that any updates to follow use the marginal
            # values.
            # XXX: This isn't really robust, but it works for now.
            mu_k = node_marginal.parents['mu']
            tau_k = node_marginal.parents['tau']


        # Divide out tau_k_gam from the (potentially marginalized)
        # node partition (i.e. node_k).
        tau_k_parts = get_linear_parts(tau_k, partial(is_, tau_k_gam))

        if isinstance(tau_k_gam, pymc.Gamma) and\
                tau_k_parts.y[0] is tau_k_gam and\
                tau_k_gam not in getattr(tau_k_parts.x[0],
                                         'extended_parents',
                                         None):
            # Normal-Gamma Update

            # Log the term we are marginalizing.
            marginal_wrt += (tau_k_gam,)

            node_marginal = norm_gamma_marginal(tau_k_gam, node_k,
                                                mu_k, tau_k)

            node_posterior = norm_gamma_post(tau_k_gam, node_k,
                                             mu_k, tau_k)

        node_marginals += ((node_marginal, node_idx, marginal_wrt),)

    # Partition updates are over, now construct the total marginal (if
    # possible).
    marginal_dist_cls = None
    from collections import defaultdict
    marginal_parent_parts = defaultdict(list)
    marginal_partitions = ()
    for node_marginal, node_idx, marginal_wrt in node_marginals:

        # We actually create/register this distribution/node now.
        marginal_partitions += (node_marginal(),)

        if marginal_dist_cls is not None and\
                marginal_dist_cls is not node_marginal.dist_cls:
            # FIXME: We do not handle multiple marginal forms.
            return
        else:
            marginal_dist_cls = node_marginal.dist_cls

        for parent_name in marginal_dist_cls.parent_names:
            parent = node_marginal.parents[parent_name]
            marginal_parent_parts[parent_name].append((parent, node_idx))

    marginal_parents = {}
    for parent_name, parent_parts in marginal_parent_parts.items():

        if len(parent_parts) > 1:
            obj_idx_lists = zip(*parent_parts)
            merged_parents = MergeIndexed(*obj_idx_lists)
        else:
            merged_parents, _ = parent_parts[0]

        marginal_parents[parent_name] = merged_parents

    marginal_parents.update({'observed': node.observed,
                             'value': node.value})
    full_marginal = marginal_dist_cls("{}-marginal".format(node.__name__),
                                      **marginal_parents)
    full_marginal._mask = node.mask
    full_marginal.partitions = marginal_partitions
    full_marginal.ismarginal = True
    full_marginal.marginalized = node
    full_marginal.marginal_wrt = pymc.SetContainer(marginal_wrt)

    marginals = getattr(node, 'marginals', tuple())
    marginals = marginals + (full_marginal,)

    node.marginals = pymc.TupleContainer(marginals)


def construct_marginal_callable(prior, node, dist_cls, parents):
    marginal_name = '{}-{}-marginal'.format(getattr(node, '__name__', ''),
                                            getattr(prior, '__name__', ''))

    kwargs = {'trace': False}

    if node.observed:
        kwargs.update({'value': node, 'observed': True})

    properties = {'_mask': getattr(node, '_mask', None),
                  'ismarginal': True,
                  'marginalized': node,
                  'marginal_wrt': pymc.SetContainer((prior,))}

    create_inst = CallableNode(dist_cls, parents, kwargs,
                               name=marginal_name,
                               properties=properties)

    return create_inst


def norm_norm_marginal(prior, node, mu_obs=None, tau_obs=None):
    r""" Create a marginal stochastic for a normal variable with a mean
    parameter consisting of a linearly transformed normally distributed
    variable.


    Parameters
    ==========
    prior: pymc.Variable, or np.ndarray
        The prior `pymc.Normal` distribution.
    mu_obs: ``prior`` or pymc.LinearCombination
        The observation/likelihood Normal mean
    tau_obs: pymc.Node
        The observation/likelihood Normal precision.
    node: pymc.Node
        The node being marginalized.

    Returns
    =======
    A function that creates the marginal distribution.
    """
    if mu_obs is None:
        mu_obs = node.parents['mu']
    if tau_obs is None:
        tau_obs = node.parents['tau']

    mu_obs_parts = get_linear_parts(mu_obs, partial(is_, prior))
    X, = mu_obs_parts.x

    mu_beta = NumpyBroadcastTo(prior.parents['mu'], prior.shape)

    mu_marginal = pymc.LinearCombination('', (X,), (mu_beta,))

    tau_beta = prior.parents['tau']

    shared_gam = set(t_ for t_ in getattr(tau_obs, 'extended_parents', set()) &
                     getattr(tau_beta, 'extended_parents', set())
                     if isinstance(t_, pymc.Gamma))

    from operator import contains
    tau_obs_parts = get_linear_parts(tau_obs, partial(contains, shared_gam))

    tau_obs_gam, = tau_obs_parts.y
    tau_obs_const, = tau_obs_parts.x

    tau_beta_parts = get_linear_parts(tau_beta, partial(is_, tau_obs_gam))
    tau_beta_gam, = tau_beta_parts.y
    tau_beta_const, = tau_beta_parts.x

    # Generally, this is what we're computing
    #
    # tau_marginal = 1/(1/tau_obs + ((X / tau_beta) * X).sum(axis=1))
    #              = tau_obs/(1 + ((X * (tau_obs / tau_beta)) * X).sum(axis=1))
    #
    # so if tau_beta = C * tau_obs,
    #
    # tau_marginal = tau_obs/(1 + ((X * (1 / C)) * X).sum(axis=1))
    #
    tau_ratio_base = 1. / tau_beta_const

    if tau_obs_gam is not tau_beta_gam:
        tau_ratio_base = tau_ratio_base * tau_obs_gam / tau_beta_gam

    X_P = X * tau_ratio_base
    X_P_X = NumpySum(X_P * X, axis=-1)
    tau_marginal_part = (1./tau_obs_const + X_P_X)**(-1)
    tau_marginal = pymc.LinearCombination('', (tau_marginal_part,),
                                          (tau_obs_gam,))

    parents = {'mu': mu_marginal, 'tau': tau_marginal}

    return construct_marginal_callable(prior, node, pymc.Normal, parents)


def norm_gamma_marginal(prior, node, mu_obs, tau_obs):
    """ Computes the normal with gamma precision prior conjugate
    marginalization.

    Parameters
    ==========
    prior: pymc.Gamma
        The gamma precision prior.
    mu_obs: pymc.Node
        Normal observation/likelihood mean.
    tau_obs: pymc.Node
        Normal observation/likelihood variance.
    node: pymc.Node
        The node being marginalized.

    Returns
    =======
    pymc.Node
        The marginal NoncentralT.
    """
    if mu_obs is None:
        mu_obs = node.parents['mu']
    if tau_obs is None:
        tau_obs = node.parents['tau']

    # Divide out prior and use the constant factor that remains.
    tau_obs_parts = get_linear_parts(tau_obs, partial(is_, prior))
    tau_obs_factor, = tau_obs_parts.x

    t_nu = 2. * prior.parents['alpha']
    t_lambda = tau_obs_factor * t_nu / prior.parents['beta']

    parents = {'mu': mu_obs, 'lam': t_lambda, 'nu': t_nu}

    return construct_marginal_callable(prior, node, pymc.NoncentralT, parents)


def norm_gamma_post(prior, node, mu_obs=None, tau_obs=None,
                    **kwargs):
    """ Computes the normal with gamma precision prior conjugate
    posterior.

    Parameters
    ==========
    prior: pymc.Gamma
        The gamma precision prior.
    node: pymc.Node
        Normal observation/likelihood.
    mu_obs: pymc.Node (optional)
        Normal likelihood mean.
    tau_obs: pymc.Node (optional)
        Normal likelihood precision.

    Returns
    =======
    pymc.Node
        The posterior pymc.Gamma.
    """
    if mu_obs is None:
        mu_obs = node.parents['mu']
    if tau_obs is None:
        tau_obs = node.parents['tau']

    # Divide out prior and use the constant factor that remains.
    tau_obs_parts = get_linear_parts(tau_obs, partial(is_, prior))
    tau_obs_factor, = tau_obs_parts.x
    assert tau_obs_parts.y[0] is prior

    # XXX: This is why having unevaluated nodes can be a confusing
    # thing: we have to check when we use one as a value.
    obs_value = node._dvalue if isinstance(node, CallableNode) else node

    r1 = obs_value - mu_obs
    r2 = pymc.LinearCombination('', (r1 * tau_obs_factor,), (r1,))

    alpha_post = prior.parents['alpha'] + NumpyAlen(obs_value)/2.
    beta_post = prior.parents['beta'] + r2/2.

    parents_post = {'alpha': alpha_post, 'beta': beta_post}

    posterior_name = "{}-post".format(prior.__name__)
    posterior = pymc.Gamma(posterior_name, **parents_post)
    posterior.isposterior = True
    posterior.posterior_wrt = node
    posterior.prior = prior
    prior.posterior = posterior
    return posterior


def norm_norm_post(beta_rv, obs, mu_obs=None, tau_obs=None, **kwargs):
    r""" Create a posterior stochastic for a normal variable that
    acts as the mean parameter for another normal variable.

    This is basically how we get the posterior mean:
    .. math:
        \beta \sim N(a, R),\quad y \sim N(F \beta, V)
        \\
        C^{-1} = F^{\top} V^{-1} F + R^{-1},\quad
        C^{-1} m = R^{-1} a + F V^{-1} y

    Parameters
    ==========
    beta_rv: pymc.Stochastic
        The prior stochastic.
    obs: pymc.Variable, or np.ndarray
        The child/observation.
    mu_obs: pymc.Variable, or np.ndarray
        A linear transform applied to ``beta_rv``
        representing :math:`y = X \beta`.
    tau_obs: pymc.Variable, or np.ndarray
        The precision for ``obs``.

    Returns
    =======
    A pymc.Normal object representing the posterior of ``beta_rv``
    given ``obs``.
    """
    if mu_obs is None:
        mu_obs = obs.parents['mu']
    if tau_obs is None:
        tau_obs = obs.parents['tau']

    if isinstance(mu_obs, pymc.LinearCombination):
        assert beta_rv is mu_obs.y[0]
        X, = mu_obs.x
    else:
        assert beta_rv is mu_obs
        X = 1

    beta_alen = np.shape(beta_rv)
    if beta_alen == ():
        beta_alen = 1
    else:
        beta_alen = beta_alen[0]
    tau_beta = beta_rv.parents['tau']
    mu_beta = beta_rv.parents['mu']

    # XXX: This is why having unevaluated nodes can be a confusing
    # thing: we have to check when we use one as a value.
    obs_value = obs._dvalue if isinstance(obs, CallableNode) else obs

    # Apply the obs's mask so that we deal with only the relevant terms.
    obs_mask = getattr(obs, '_mask', None)
    if obs_mask is not None:
        X = X[obs_mask]
        obs_value = obs_value[obs_mask]

    def rhs(t_b_=tau_beta, m_b_=mu_beta, X_=X, t_y_=tau_obs,
            y_=obs_value, b_a_=beta_alen):
        X_ = np.broadcast_to(X_, (np.alen(y_), b_a_))
        m_b_ = np.broadcast_to(m_b_, np.shape(X_)[-1:])
        t_b_ = np.broadcast_to(t_b_, np.shape(X_)[-1:])
        #res = np.dot(t_b_, m_b_) + np.dot(X_.T * t_y_, y_)
        rhs_res = t_b_ * m_b_ + np.dot(np.transpose(X_) * t_y_, np.ravel(y_))
        return np.ravel(rhs_res)

    rhs = pymc.Lambda('{}-rhs'.format(beta_rv.__name__), rhs,
                      trace=False)

    def tau_update(t_b_=tau_beta, X_=X, t_y_=tau_obs,
                   y_=obs_value, b_a_=beta_alen):
        X_ = np.broadcast_to(X_, (np.alen(y_), b_a_))
        t_b_ = np.broadcast_to(t_b_, np.shape(X_)[-1:])
        t_y_ = np.broadcast_to(t_y_, np.alen(y_))
        #res = t_b_ + np.dot(X_.T * t_y_, X_)
        X_t_y = np.einsum('ij,i->ij', X_, np.atleast_1d(t_y_))
        tau_res = t_b_ + np.sum(X_t_y * X_, axis=0)
        return tau_res

    tau_post = pymc.Lambda('{}-tau-post'.format(beta_rv.__name__),
                           tau_update, trace=False)

    def mu_update(t_p_=tau_post, rhs_=rhs):
        #res = np.linalg.solve(t_p_, rhs_)
        res = rhs_/t_p_
        return res

    mu_post = pymc.Lambda('{}-mu-post'.format(beta_rv.__name__),
                          mu_update, trace=False)

    beta_post = pymc.Normal("{}-post".format(beta_rv.__name__),
                            mu_post, tau_post, **kwargs)

    beta_post.isposterior = True
    beta_post.posterior_wrt = obs
    beta_post.prior = beta_rv
    beta_rv.posterior = beta_post

    return beta_post
