import numpy as np 
from GPy.models import GPKroneckerGaussianRegression
from GPy.kern import *
from itertools import izip
import h5py
from scipy.linalg inv
from multiprocess import Pool
import emcee as mc

def lnprior(theta):

    om, s8 = theta

    if om < MIN_OM or om > MAX_OM:
        return -np.inf
    if s8 < MIN_S8 or s8 > MAX_S8:
        return -np.inf
    return 0

def lnlike(theta, nu, y, inv_cov):
    """
    :param theta:
        Proposed parameters.
    :param param_names:
        The names of the parameters in theta
    :param fixed_params:
        Dictionary of parameters necessary to predict y_bar but are not being sampled over.
    :param r_bin_centers:
        The centers of the r bins y is measured in, angular or radial.
    :param ys:
        The measured values of the observables to compare to the emulators. Must be an interable that contains
        predictions of each observable.
    :param combined_inv_cov:
        The inverse covariance matrices. Explicitly, the inverse of the sum of the mesurement covaraince matrix
        and the matrix from the emulator, both for each observable. Both are independent of emulator parameters,
         so can be precomputed. Must be an iterable with a matrixfor each observable.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    """

    emu_pred = 10 ** emu.predict(theta.reshape((1,-1)),nu, mean_only=True)[0]

    delta = emu_pred - y
    #print delta
    return - np.dot(delta, np.dot(inv_cov, delta))

def lnprob(theta, *args):
    """
    The total liklihood for an MCMC. Mostly a generic wrapper for the below functions.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    """
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, *args)

def run_mcmc_iterator(y, cov, nu, pos0, nwalkers=1000, nsteps=100, ncores=16):
    """
    Run an MCMC using emcee and the emu. Includes some sanity checks and does some precomputation.
    Also optimized to be more efficient than using emcee naively with the emulator.

    This version, as opposed to run_mcmc, "yields" each step of the chain, to write to file or to print.

    :param emus:
        A trained instance of the Emu object. If there are multiple observables, should be a list. Otherwiese,
        can be a single emu object
    :param param_names:
        Names of the parameters to constrain
    :param y:
        data to constrain against. either one array of observables, of size (n_bins*n_obs)
    # TODO figure out whether it should be row or column and assign appropriately
    :param cov:
        measured covariance of y for each y. Should have the same shape as y, but square
    :param r_bin_centers:
        The scale bins corresponding to all y in ys
    :param resume_from_previous:
        String listing filename of a previous chain to resume from. Default is None, which starts a new chain.
    :param fixed_params:
        Any values held fixed during the emulation, default is {}
    :param nwalkers:
        Number of walkers for the mcmc. default is 1000
    :param nsteps:
        Number of steps for the mcmc. Default is 1--
    :param nburn:
        Number of burn in steps, default is 20
    :param ncores:
        Number of cores. Default is 'all', which will use all cores available
    :param return_lnprob:
        Whether to return the evaluation of lnprob on the samples along with the samples. Default is Fasle,
        which only returns samples.
    :yield:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """

    ncores = ncores#_run_tests(y, cov, r_bin_centers, param_names, fixed_params, ncores)
    pool = Pool(processes=ncores)

    inv_cov = inv(cov)

    sampler = mc.EnsembleSampler(nwalkers, 2, lnprob, pool=pool,
                                 args=(nu, y, inv_cov) )

    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        yield result[0]

if __name__ == '__main__':
    training_pc = np.load('training_pc.npy')
    training_cov = np.load('training_cov.npy')
    training_err = np.sqrt(np.diag(training_cov))

    test_pc = np.load('test_pc.npy')
    test_cov = np.load('test_cov.npy')
    test_err = np.sqrt(np.diag(test_cov))

    training_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTraining/UatuFastPMTraining.hdf5'
    test_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTest/UatuFastPMTest.hdf5'

    train_cosmos = np.zeros((training_pc.shape[0], 2))
    test_cosmos = np.zeros((test_pc.shape[0], 2))

    with h5py.File(training_filename, 'r') as f:
        for i, key in enumerate(f.keys()):
            train_cosmos[i] = f[key]['Y'][0].squeeze()

    with h5py.File(test_filename, 'r') as f:
        for i, key in enumerate(f.keys()):
            test_cosmos[i] = f[key]['Y'][0].squeeze()

    mean_cosmo = test_cosmos.mean(axis=0)

    test_idx = np.argmin((test_cosmos-mean_cosmo)**2, axis = 0)

    y = test_pc[test_idx]
    cov = test_cov[test_idx]

    # set prior bounds
    MIN_OM, MIN_S8 = train_cosmos.min(axis=0)
    MAX_OM, MAX_S8 = train_cosmos.max(axis=0)

    global MIN_OM, MIN_S8, MAX_S8, MAX_OM
    thresholds = np.linspace(-0.01, 0.04, 41)
    nu = ((thresholds[1:] + thresholds[:-1]) / 2.0).reshape((-1, 1))

    kern1 = RBF(2, ARD=True)
    kern2 = RBF(1, ARD=True)

    emu = GPKroneckerGaussianRegression(train_cosmos, nu, np.log10(training_pc), kern1,
                                       kern2)  # , Yerr=np.log10(training_err))
    emu.optimize_restarts(num_restarts=5, verbose = True);
    global emu
    nwalkers, nsteps = 500, 2000

    pos0 = np.random.randn(nwalkers, 2)
    pos0[:,0] = pos0[:,0]*0.1 +0.3
    pos0[:,1] = pos0[:,1]*0.1+0.8
    chain_fname = '~/scratch/uatu_preds/uatu_pc_emu_mcmc.hdf5'
    with h5py.File(chain_fname, 'w') as f:
        f.create_dataset('chain', (0, 2), chunks = True, compression = 'gzip', maxshape = (None, 2))

    for step, pos in enumerate(run_mcmc_iterator(y, cov, nu,  nwalkers=nwalkers, \
                                                 nsteps=nsteps, ncores=16,
                                                 pos0=pos0)):
        with h5py.File(chain_fname, 'r+') as f:
        # f.swmr_mode = True
            chain_dset = f['chain']
            l = len(chain_dset)
            chain_dset.resize((l + nwalkers), axis=0)
            chain_dset[-nwalkers:] = pos[0]
