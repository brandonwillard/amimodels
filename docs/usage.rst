Usage
=====

The current way to use these objects is to create/obtain a vector of observations,
``y``, and a list of design matrices/features, ``X_matrices``, as Pandas
DataFrames or numpy.ndarrays.  The length of the list implies the number
of mixture components/states in the model, and the individual matrices
determine the regression terms estimated in each state.  If no exogenous
terms are included, i.e. the matrix is a column of ones, then only a single constant
term is estimated.

Next, one can run an initialization function to obtain initial values for the
model parameters, then use one of the model generation functions to produce the
PyMC objects.  Given PyMC objects, the model can be sampled or used to estimate
a MAP value (see the
`PyMC2 documentation <https://pymc-devs.github.io/pymc/modelfitting.html>`_).

Here is an example workflow::

    from amimodels.normal_hmm import *
    from amimodels.step_methods import *

    demo_process = NormalHMMProcess(np.asarray([[0.9], [0.1]]),
                                    200,
                                    np.asarray([1., 0.]),
                                    np.asarray([[0.2], [1.]]),
                                    np.asarray([0.1/8.**2, 0.5/3.**2]),
                                    seed=249298)

    # Produce a simulation
    states_true, y, X_matrices = demo_process.simulate()

    # Construct some intial parameters to an HMM
    init_params = gmm_norm_hmm_init_params(y, X_matrices)

    # Build the model using our initial params.
    norm_hmm = make_normal_hmm(y, X_matrices, init_params)

    norm_mcmc = pymc.MCMC(norm_hmm.variables)

    # If you want to use our special sampling step methods...
    norm_mcmc.use_step_method(HMMStatesStep,
                              norm_hmm.states)
    norm_mcmc.use_step_method(TransProbMatStep,
                              norm_hmm.trans_mat)

    norm_mcmc.sample(1000)


For more examples, see this packages' :file:`tests` and :file:`examples`
folders.

