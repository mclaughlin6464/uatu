from uatu.watchers import *
from sklearn.model_selection import train_test_split
import h5py
from time import time

dir = '/scratch/users/swmclau2/UatuLightconeTest/'
fname = path.join(dir, 'UatuLightconeTest.hdf5')

test_dset = DatasetFromFile(fname, batch_size=32, shuffle = False, augment = False, whiten = False, train_test_split = 1.0, cache_size = 20)

#target_y = np.array([0.26, 0.76]).reshape((1, 2))
target_y = np.array([0.311, 0.909]).reshape((1, 2))

print target_y

target_fname = path.join(dir, 'UatuLightconeAttackedTranspose.hdf5')

f = h5py.File(fname, 'r')
attrs = dict()
for key in f.attrs:
    attrs[key] = f.attrs[key]
f.close()

compute_attacked_maps(gupta_network_init_fn, standard_cost_fn, \
        '/home/users/swmclau2/scratch/uatu_networks/gupta_net_kappa-28900', test_dset, target_y,\
        target_fname, attrs)

