from uatu.watchers import *
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
from time import time

dir = '/scratch/users/swmclau2/UatuLightconeTest/'
fname = path.join(dir, 'UatuLightconeTest.hdf5')

test_dset = DatasetFromFile(fname, batch_size=1, shuffle = False, augment = False, whiten = True, train_test_split = 1.0, cache_size = 100)

#target_y = np.array([0.26, 0.76]).reshape((1, 2))
#target_y = np.array([0.311, 0.909]).reshape((1, 2))

# build the true-to_target map
f = h5py.File(fname, 'r')
true_ys = np.zeros((50, 2))
for idx, key in enumerate(sorted(f.keys())):
    true_ys[idx] = f[key]['Y'][0].squeeze()

true_to_target_map = dict()

shuffled_idxs = np.random.choice(true_ys.shape[0], true_ys.shape[0], replace = False)

for target_idx, ty in zip(shuffled_idxs, true_ys):
    true_to_target_map[tuple(ty)] = true_ys[target_idx]

target_fname = path.join(dir, 'UatuLightconeAttackedShuffled.hdf5')

f = h5py.File(fname, 'r')
attrs = dict()
for key in f.attrs:
    attrs[key] = f.attrs[key]
f.close()

compute_shuffled_attacked_maps(gupta_network_init_fn, standard_cost_fn, \
        '/home/users/swmclau2/scratch/uatu_networks/gupta_net_kappa-28900', test_dset, target_y,\
        target_fname, attrs)

