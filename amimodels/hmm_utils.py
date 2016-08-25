import numpy as np


def compute_trans_freqs(states, N_states, counts_only=False):
    """ Computes empirical state transition frequencies.

    Parameters
    ==========
    states: a pymc object or ndarray
        Vector sequence of states.
    N_states: int
        Total number of observable states.
    counts_only: boolean
        Return only the transition counts for each state.

    Returns
    =======
        Unless `counts_only` is `True`, return the empirical state transition
        frequencies; otherwise, return the transition counts for each state.
    """
    states_ = getattr(states, 'values', states).ravel()

    if any(np.isnan(states_)):
        states_ = np.ma.masked_invalid(states_).astype(np.uint)
        states_mask = np.ma.getmask(states_)
        valid_pairs = ~states_mask[:-1] & ~states_mask[1:]
        state_pairs = (states_[:-1][valid_pairs], states_[1:][valid_pairs])
    else:
        state_pairs = (states_[:-1], states_[1:])

    counts = np.zeros((N_states, N_states))
    flat_coords = np.ravel_multi_index(state_pairs, counts.shape)
    counts.flat += np.bincount(flat_coords, minlength=counts.size)
    counts = np.nan_to_num(counts)

    if counts_only:
        return counts

    freqs = counts/np.maximum(1, counts.sum(axis=1, keepdims=True))
    return freqs


def compute_steady_state(trans_mat):
    """ Compute the steady state of a transition probability matrix.

    Parameters
    ==========

    trans_mat: numpy.array
        A transition probability matrix for K states with shape
        (K, K-1), i.e. the last column omitted.

    Returns
    =======

    A numpy.array representing the steady state.
    """

    N_states = trans_mat.shape[0]
    if not (trans_mat.ndim == 2 and trans_mat.shape[1] == N_states - 1):
        raise ValueError("trans_mat must have ndim == 2 and shape (K, K-1)")

    P = np.column_stack((trans_mat, 1. - trans_mat.sum(axis=1)))
    Lam = (np.eye(N_states) - P + np.ones((N_states, N_states))).T
    u = np.linalg.solve(Lam, np.ones(N_states))
    return u
