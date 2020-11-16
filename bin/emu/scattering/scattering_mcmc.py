import numpy as np 
from GPy.models import GPKroneckerGaussianRegression
from GPy.kern import *
from itertools import izip
import h5py
from scipy.linalg import inv
from multiprocessing import Pool
import emcee as mc
import cPickle as pickle

def lnprior(theta, *args):

    om, s8 = theta
    if om < MIN_OM or om > MAX_OM:
        return -np.inf
    if s8 < MIN_S8 or s8 > MAX_S8:
        return -np.inf
    return 0

def lnlike(theta, Js, y, inv_cov, use_idxs):
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
    emu_preds = [] 
    for emu in emus:
        if len(emu.Y.shape)>2:
            emu_pred = emu.predict(theta.reshape((1,-1)),Js, mean_only=True,
                     additional_Xnews=[Js])[0].squeeze()
            emu_pred = emu_pred.reshape((J,J)).T.flatten() # need to reshape
        else:
            emu_pred = emu.predict(theta.reshape((1,-1)),Js, mean_only=True)[0].squeeze()
        emu_preds.append(emu_pred)
    
    emu_pred = np.exp(np.hstack(emu_preds)[use_idxs])
    #print theta
    #print emu_pred 
    #print y
    delta = emu_pred - y
    #print - np.dot(delta, np.dot(inv_cov, delta))

    #print delta
    #print '*'*10

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

def run_mcmc_iterator(y, cov, Js, use_idxs, pos0, nwalkers=1000, nsteps=100, ncores=16):
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

    sampler = mc.EnsembleSampler(nwalkers, 2, lnprob, #pool=pool,
                                 args=(Js, y, inv_cov, use_idxs) )

    #print lnlike(np.array([[ 0.27992499,  0.84698802]]), Js, y, inv_cov)
    for result in sampler.sample(pos0, iterations=nsteps):#, storechain=False):
        yield result#[0]

if __name__ == '__main__':

    smooth = 1.0
    noise = 0.29
    J = 4

    training_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTraining/UatuFastPMTrainingScattering_smooth_%0.1f_noise_%0.1f.hdf5'%(smooth,noise)
    test_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTest/UatuFastPMTestScattering_smooth_%0.1f_noise_%0.1f.hdf5'%(smooth,noise)

    train_cosmos = np.zeros((200, 2))
    test_cosmos = np.zeros((50, 2))

    train_scattering = np.zeros((200, 1296, 21))
    test_scattering = np.zeros((50, 1296, 21))
    with h5py.File(training_filename, 'r') as f:
        for i, key in enumerate(f.keys()):
            train_cosmos[i] = f[key]['Y'][0].squeeze()

            train_scattering[i] = f[key]['X'][()]

    skip_idx = 9
    idxs = np.ones((200,), dtype=bool)
    idxs[skip_idx] = False
    train_scattering = train_scattering[idxs]
    train_cosmos = train_cosmos[idxs]

    X1_train = np.log(train_scattering[:,:, 1:1+J])
    X2_train = np.log(train_scattering[:,:,1+J:])#- np.repeat(X1_train, J, axis=2)
    X_train = np.concatenate([X1_train, X2_train], axis =2 )

    X_train_bar = X_train.mean(axis=1)
    #cov = np.mean(np.stack([np.cov(_X, rowvar=False) for _X in X_train]), axis =0)
    #cov = np.mean(np.stack([np.cov(_X, rowvar=False) for _X in np.exp(X_train)]), axis =0)

    with h5py.File(test_filename, 'r') as f:
        for i, key in enumerate(f.keys()):
            test_cosmos[i] = f[key]['Y'][0].squeeze()

            test_scattering[i] = f[key]['X'][()]

    X1_test = np.log(test_scattering[:,:, 1:1+J])
    X2_test = np.log(test_scattering[:,:,1+J:])#- np.repeat(X1_test, J, axis=2)
    X_test = np.concatenate([X1_test, X2_test], axis =2 )

    X_test_bar = X_test.mean(axis=1)

    #print X_train.mean(axis=(0,1))
    #print X_test.mean(axis=(0,1))

    mean_cosmo = test_cosmos.mean(axis=0)
    test_idx = np.argmin(np.sum((test_cosmos-mean_cosmo)**2, axis = 1), axis = 0)
    y = np.exp(X_test_bar[test_idx])
    cov = np.cov(np.exp(X_test[test_idx]), rowvar=False)
    # set prior bounds

    global MIN_OM, MIN_S8, MAX_S8, MAX_OM
    MIN_OM, MIN_S8 = train_cosmos.min(axis=0)
    MAX_OM, MAX_S8 = train_cosmos.max(axis=0)

    Js = np.array(range(J)).reshape((-1,1))
    #with open('scattering_emu_kern_noise_%0.1f_smooth_%0.1f.pkl'%(noise,smooth), 'r') as f:
    #        kerns = pickle.load(f)
    #global emus
    #emus = []
    #klist = []
    #for emu in kerns:
    #    klist.append([])
    kern1 = RBF(2, ARD=True)+Bias(2)
    kern2 = RBF(1, ARD=True)+Bias(1)
    kern3 = RBF(1, ARD=True)+Bias(1)

    #    for i, (k, ko) in enumerate(zip(emu, [kern1,kern2,kern3])):
            #print ko.from_dict(k)
    #        klist[-1].append(ko.from_dict(k))

    emu1 = GPKroneckerGaussianRegression(train_cosmos, Js, X1_train.mean(axis=1), kern1,
                                       kern2 , Yerr=X1_train.std(axis=1)**2)
    kern1 = RBF(2, ARD=True)+Bias(2)
    kern2 = RBF(1, ARD=True)+Bias(1)
    kern3 = RBF(1, ARD=True)+Bias(1)


    emu2 = GPKroneckerGaussianRegression(train_cosmos, Js, X2_train.mean(axis=1).reshape((-1,J,J)), kern1,
                                       kern2, additional_Xs=[Js], additional_kerns=[kern3] , Yerr=(X2_train.std(axis=1)**2).reshape((-1,J,J)) )
    emus = [emu1,emu2]

    for emu in emus:
        emu.optimize_restarts(num_restarts=5, robust=True)
    
    #emus = [emu1]
    #y = y[:J]
    #cov = cov[:J][:,:J]
    # drop the parts with no information
    use_idxs = np.zeros((20,), dtype=bool)
    use_idxs[:J] = True  #s1
    #use_idxs[J:] = True #s2
    for i in xrange(J): #s2, but with non-informative parts removed
        j = i+1
        use_idxs[(i+1)*J+j:(i+2)*J] = True
    y = y[use_idxs]
    #cov = np.diag(np.diag(cov)[use_idxs])
    cov = cov[use_idxs][:, use_idxs]
    #emu.optimize_restarts(num_restarts=5, verbose = True);
    nwalkers, nsteps = 100, 1000 

    pos0 = np.random.randn(nwalkers, 2)
    pos0[:,0] = pos0[:,0]*0.1 +0.3
    pos0[:,1] = pos0[:,1]*0.1+0.8
    if sum(use_idxs)==J:
        chain_fname = '/scratch/users/swmclau2/uatu_preds/uatu_scattering_smooth_%0.1f_noise_%0.1f_s1_emu_mcmc.hdf5'%(smooth,noise)
    elif sum(use_idxs)==J*(J-1)/2:
        chain_fname = '/scratch/users/swmclau2/uatu_preds/uatu_scattering_smooth_%0.1f_noise_%0.1f_s2_emu_mcmc.hdf5'%(smooth,noise)

    else:
        chain_fname = '/scratch/users/swmclau2/uatu_preds/uatu_scattering_smooth_%0.1f_noise_%0.1f_s1_s2_emu_mcmc.hdf5'%(smooth,noise)

    print 'Truth', test_cosmos[test_idx]

    with h5py.File(chain_fname, 'w') as f:
        f.create_dataset('chain', (0, 2), chunks = True, compression = 'gzip', maxshape = (None, 2))

    for step, pos in enumerate(run_mcmc_iterator(y, cov, Js, use_idxs,  nwalkers=nwalkers, \
                                                 nsteps=nsteps, ncores=16,
                                                 pos0=pos0)):
        with h5py.File(chain_fname, 'r+') as f:
        # f.swmr_mode = True
            chain_dset = f['chain']
            l = len(chain_dset)
            chain_dset.resize((l + nwalkers), axis=0)
            #print pos.coords
            chain_dset[-nwalkers:] = pos[0]#.coords#[0]
