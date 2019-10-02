from uatu.watchers import simulate_analysis_iterator 
from scipy.linalg import inv
import numpy as np

boxno = 0
obsboxdir = '/home/users/swmclau2/scratch/UatuTest4/Box%03d/'%boxno

bins = np.loadtxt('/home/users/swmclau2/scratch/uatu_bins.npy')

cov = np.loadtxt('/home/users/swmclau2/scratch/uatu_cov.npy')

invcov = inv(cov)

nwalkers = 10 
nsteps = 500 

chain_fname = '/home/users/swmclau2/scratch/simulated_chain4_%02d_walkers_%02d_steps.npy'%(nwalkers, nsteps)

with open(chain_fname, 'w') as f:
    f.write('#' + '\t'.join(['Om', 's8', 'log_sOm', 'log_ss8', 'rho'])+'\n')

for pos in simulate_analysis_iterator(obsboxdir, bins, invcov, nwalkers, nsteps, ncores=1):

    with open(chain_fname, 'a') as f:
        np.savetxt(f, pos)

