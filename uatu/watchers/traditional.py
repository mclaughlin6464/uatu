from __future__ import print_function

"""
Perform a traditional analysis on the matter correlation function.
"""
from os import path
from glob import glob
import numpy as np
try:
    from nbodykit.source.catalog import CSVCatalog
    from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
    from nbodykit.cosmology.correlation import CorrelationFunction
    from nbodykit.cosmology.power.halofit import HalofitPower
    from nbodykit.cosmology import Cosmology
except:
    pass # nbodykit can be finnicky
import emcee as mc

def compute_xi_from_box(bins, boxdir):
    names  = ['x', 'y','z','vx','vy','vz']
    f = CSVCatalog(path.join(boxdir, 'uatu_z0p000.0'), names)

    downsample_idxs = np.random.choice(f['x'].shape[0], int(1e-3*f['x'].shape[0]), replace = False)
    d_idxs = np.zeros((f['x'].shape[0],))
    d_idxs[downsample_idxs] = 1
    f = f[d_idxs.astype(bool)]
    f['Position'] = np.c_[f['x'], f['y'], f['z']]
    s = SimulationBox2PCF('1d', f, bins, BoxSize = 512.0, show_progress = True)

    return np.array([x[0] for x in s.corr.data])

def compute_cov_from_all_boxes(bins, dir, max = 50):

    assert path.isdir(dir)
    xis = []
    all_subdirs = glob(path.join(dir, 'Box*/'))
    for boxno, subdir in enumerate(sorted(all_subdirs)):
        print(subdir)
        if boxno == max:
            break
        xis.append(compute_xi_from_box(bins, subdir))

    xis = np.array(xis)
    print(type(xis), xis.shape)
    print(xis.dtype)
    print(xis)

    return np.cov(xis, rowvar = False)#, axis = 0)

def lnprior(theta):
    Om, s8 = theta
    if Om < 0.2 or Om > 0.4:
        return -np.inf
    if s8 <0.75 or s8 > 1.1:
        return -np.inf
    return 0

def lnliklihood(theta, bins, obs_xi, invcov):

    Om, s8 = theta
    Om*=0.7**2
    Ob = 0.022
    Ocdm = Om - Ob
    cosmo = Cosmology(h=1.0, T0_cmb=2.726,
                      Omega_b = Ob,
                      Omega0_cdm=Ocdm,
                      P_k_max = 100.0,
                      n_s=0.96).match(s8)
    bc = (bins[1:] + bins[:-1])/2.0
    predicted_xi = CorrelationFunction(HalofitPower(cosmo, 0.0))(bc)

    delta =  predicted_xi - obs_xi
    return -0.5*np.sum(np.dot(delta, np.dot(invcov, delta)))

def lnprob(theta, *args):
    lp = lnprior(theta)
    if np.isfinite(lp):
        ll =lnliklihood(theta, *args)
        return lp + ll
    return lp


def simulate_analysis(obsboxdir, bins, invcov, nwalkers, nsteps, ncores):

    assert bins.shape[0]-1 == invcov.shape[0]
    obs_xi = compute_xi_from_box(bins, obsboxdir)

    sampler = mc.EnsembleSampler(nwalkers, 2, lnprob,
                                 threads=ncores, args=(bins, obs_xi, invcov))

    pos0 = np.random.randn(nwalkers, 2)
    pos0*=0.05
    pos0+= np.array([0.3, 0.9])
    sampler.run_mcmc(pos0, nsteps)

    return sampler.flatchain

def simulate_analysis_iterator(obsboxdir, bins, invcov, nwalkers, nsteps, ncores):

    assert bins.shape[0]-1 == invcov.shape[0]
    obs_xi = compute_xi_from_box(bins, obsboxdir)

    sampler = mc.EnsembleSampler(nwalkers, 2, lnprob,
                                 threads=ncores, args=(bins, obs_xi, invcov))

    pos0 = np.random.randn(nwalkers, 2)
    pos0*=0.05
    pos0+= np.array([0.3, 0.9])

    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        yield result[0]
