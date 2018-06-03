import nbodykit
from uatu.watchers.traditional import *
import numpy as np
from time import time

t0 = time()
fname = '/home/users/swmclau2/scratch/UatuTraining4/'

bins = np.logspace(-0.5, 1.6, 10)
np.savetxt('/home/users/swmclau2/scratch/uatu_bins.npy', bins)

cov = compute_cov_from_all_boxes(bins, fname)

print time() - t0
np.savetxt('/home/users/swmclau2/scratch/uatu_cov.npy', cov)
