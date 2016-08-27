import numpy as np
import pandas as pd

import pymc


def plot_hmm(mcmc_step, obs_index=None, axes=None,
             smpl_subset=None, states_true=None):
    r""" Plot the observations, estimated observation mean parameter's statistics
    and estimated state sequence statistics.

    Parameters
    ==========
    mcmc_res: a pymc.MCMC object
        The MCMC object after estimation.
    obs_index: pandas indices or None
        Time series index for observations.
    axes: list of matplotlib axes
        Axes to use for plotting.
    smpl_subset: float, numpy.ndarray of int, or None
        If a float, the percent of samples to plot; if a `numpy.ndarray`,
        the samples to plot; if None, plot all samples.
    states_true: pandas.DataFrame
        True states time series.

    Returns
    =======
    A matplotlib plot object.

    """
    if axes is None:
        from matplotlib.pylab import plt
        fig, axes = plt.subplots(2, 1)

    _ = [ax_.clear() for ax_ in axes]

    #start_idx = y_baseline.index.searchsorted("2014-11-23 00:00:00")
    #end_idx = y_baseline.index.searchsorted("2014-12-01 00:00:00")
    #range_slice = slice(start_idx, end_idx)

    y_rv, = mcmc_step.observed_stochastics
    y_df = pd.DataFrame(y_rv.value, index=obs_index, columns=[r"$y_t$"])
    y_df.plot(ax=axes[0], drawstyle='steps-mid', zorder=3)

    # Would be nice to have a y_rv.mean, or something.
    if isinstance(y_rv, pymc.Normal):
        mu_rv = y_rv.parents['mu']
    elif len(y_rv.parents) == 1:
        mu_rv, = y_rv.parents.values()
    else:
        raise ValueError("Unsupported observation distribution")

    N_samples = mcmc_step._iter
    if smpl_subset is not None:
        if isinstance(smpl_subset, float):
            smpls_idx = np.random.randint(0, N_samples,
                                          size=int(N_samples * smpl_subset))
        else:
            smpls_idx = smpl_subset
    else:
        smpls_idx = xrange(N_samples)

    mu_samples = mcmc_step.trace(mu_rv.__name__).gettrace()[smpls_idx].T
    mu_df = pd.DataFrame(mu_samples, index=y_df.index)

    mu_df.plot(ax=axes[0], zorder=1, drawstyle='steps-mid',
               legend=False, color='lightgreen',
               alpha=5. / len(smpls_idx), label=None,
               rasterized=True)
    mu_mean_df = pd.DataFrame(mcmc_step.trace(mu_rv.__name__).gettrace().mean(axis=0),
                              index=y_df.index,
                              columns=['$E[\mu_t \mid y_{1:T}]$'])
    mu_mean_df.plot(ax=axes[0], zorder=1, drawstyle='steps-mid',
                    color='green', alpha=0.5,
                    fillstyle=None)

    # TODO: Add mu_true and posterior predictives.

    leg_top = axes[0].get_legend()
    leg_top.get_frame().set_alpha(0.7)

    if states_true is not None:
        #states_true_df = pd.DataFrame(states_true, index=y_df.index,
        #                              columns=["$S_t$ true"]) + 1
        states_true_df = states_true + 1
        states_true_df.columns = ["$S_t$ true"]
        states_true_df.plot(ax=axes[1], zorder=2, alpha=0.5,
                            drawstyle='steps-mid', linewidth=3,
                            linestyle="dashed")

    from amimodels.stochastics import HMMStateSeq
    states_rv, = filter(lambda x: isinstance(x, HMMStateSeq),
                        mcmc_step.stochastics)

    state_samples = states_rv.trace.gettrace()[smpls_idx]
    states_df = pd.DataFrame(state_samples.T, index=y_df.index) + 1
    states_df.plot(ax=axes[1], zorder=1, alpha=0.5,
                   drawstyle='steps-mid', legend=False,
                   color='lightgreen', label=None)
    states_mean_df = pd.DataFrame(states_rv.trace.gettrace().mean(axis=0) + 1,
                                  index=y_df.index,
                                  columns=['$E[S_t \mid y_{1:T}]$'])
    states_mean_df.plot(ax=axes[1], zorder=2,
                        drawstyle='steps-mid',
                        alpha=0.5, color='green')

    axes[1].set_yticks(range(1, states_rv.K+1))
    axes[1].set_ybound((0.8, states_rv.K + 0.2))

    leg_bottom = axes[1].get_legend()
    leg_bottom.get_frame().set_alpha(0.7)

    fig = getattr(axes[0], 'figure', None)
    if fig is not None:
        fig.tight_layout()

    return axes


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
