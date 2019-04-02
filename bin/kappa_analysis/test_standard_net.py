from uatu.watchers import *
from sklearn.model_selection import train_test_split
from time import time

dir = '/scratch/users/swmclau2/UatuLightconeTest/'
fname = path.join(dir, 'UatuLightconeTest.hdf5')

test_dset = DatasetFromFile(fname, batch_size=1, shuffle = False, augment = False, whiten = True, train_test_split = 1.0, cache_size = 1)

test(gupta_network_init_fn, test_dset, n_samples = 1 ,\
        fname = '/home/users/swmclau2/scratch/gupta_net_kappa-28900', samples_fname = path.join(dir, 'UatuLightconePreds.hdf5')) 

