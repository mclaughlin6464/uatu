from uatu.watchers import simulate_analysis 
from scipy.linalg import inv
import numpy as np

boxno = 0
obsboxdir = '/home/users/swmclau2/scratch/UatuTest4/Box%03d/'%boxno

bins = np.loadtxt('/home/users/swmclau2/scratch/uatu_bins.npy')

cov = np.loadtxt('/home/users/swmclau2/scratch/uatu_cov.npy')

invcov = inv(cov)

nwalkers = 100 
nsteps = 500 

chain = simulate_analysis(obsboxdir, bins, invcov,nwalkers, nsteps, ncores = 8) 

np.savetxt('/home/users/swmclau2/scratch/simulated_chain_%02d_walkers_%02d_steps.npy'%(nwalkers, nsteps), chain)
