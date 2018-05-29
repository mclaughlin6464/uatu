"""
Perform a traditional analysis on the matter correlation function.
"""
from os import path
from glob import glob
import numpy as np
from nbodykit.source.catalog import CSVCatalog
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit.cosmology.correlation import CorrelationFunction
from nbodykit.cosmology.power.halofit import HalofitPower
from nbodykit.cosmology import Cosmology
import emcee as mc

def compute_xi_from_box(bins, boxdir):
    names  = ['x', 'y','z','vx','vy','vz']
    f = CSVCatalog(path.join(boxdir, 'uatu_z0p000.0'), names)

    s = SimulationBox2PCF('1d', f, bins)

    return s.corr

def compute_cov_from_all_boxes(bins, dir):

    assert path.isdir(dir)
    xis = []
    all_subdirs = glob(path.join(dir, 'Box*/'))
    for boxno, subdir in enumerate(sorted(all_subdirs)):
        print subdir
        xis.append(compute_xi_from_box(bins, subdir))

    xis = np.array(xis)

    return np.cov(xis, axis = 0)

def lnprior(theta):
    Om, s8 = theta
    if Om < 0 or Om > 1:
        return -np.inf
    if s8 <0 or s8 > 2:
        return -np.inf
    return 0

def lnliklihood(theta, bins, obs_xi, invcov):

    Om, s8 = theta
    Ocdm = Om - 0.022
    cosmo = Cosmology(h=0.7, T0_cmb=2.726,
                      Omega_b = 0.022,
                      Omega0_cdm=Ocdm,
                      n_s=0.96).match(s8, Om)
    predicted_xi = CorrelationFunction(HalofitPower(cosmo, 0.0))(bins)
    delta =  predicted_xi - obs_xi
    return -0.5*np.sum(np.dot(delta, np.dot(invcov, delta)))

def lnprob(theta, *args):
    lp = lnprior(theta)
    if np.isfinite(lp):
        ll =lnliklihood(theta, *args)
        return lp + ll
    return lp


def simulate_analysis(obsboxdir, bins, invcov, nwalkers, nsteps, ncores):

    assert bins.shape[0] == invcov.shape[0]
    obs_xi = compute_xi_from_box(bins, obsboxdir)

    sampler = mc.EnsembleSampler(nwalkers, 2, lnprob,
                                 threads=ncores, args=(bins, obs_xi, invcov))

    pos0 = np.random.randn(nwalkers, 2)
    pos0*=0.05
    pos0+= np.array([0.25, 0.8])
    sampler.run_mcmc(pos0, nsteps)

    return sampler.chain
