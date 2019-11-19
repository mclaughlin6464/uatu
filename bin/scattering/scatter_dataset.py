# Just scatter a whole dataset, so we can do PCA, etc, on them.

from kymatio import Scattering2D
import torch
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func


mode = 2
max_order = 2 if mode == 2 else 1
J = 2 # 0, 1, or
shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

scattering = Scattering2D(J=J, shape=shape, max_order=max_order)
if use_cuda:
    scattering = scattering.cuda()

L = 8
# K = 1 + L*J+ (L**2)*(J*(J-1))/2.0

if mode == 0:
    # do a little fuckery
    _scattering = scattering
    scattering = lambda x: _scattering(x)[:, 0]
    K = 1
if mode == 1:
    K = 1 + L * J
    # K = L*J
    # _scattering = scattering
    # scattering = lambda x: _scattering(x)[:,1:]

elif mode == 2:
    K = int(1 + L * J + (L ** 2) * (J * (J - 1)) / 2.0)
    # K = int(L*J + (L**2)*(J*(J-1))/2.0)
    # _scattering = scattering
    # scattering = lambda x: _scattering(x)[:,1:]

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

output_fname  = path.join(dir, 'UatuLightconeTrainingScattered.hdf5')

batch_size = 1
train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True,
                             train_test_split = 1.0, whiten = True, cache_size = 289, transform=torch.Tensor)

key_dict = {}
with h5py.File(path.join(dir, output_fname), 'w') as f:
    for i, (x,y) in enumerate(train_dset):
        print(i, y)
        scatter = scattering(x.squeeze().cuda()).cpu().numpy()

        key = key_func(y.reshape((1, 2)))
        if key not in key_dict:
            key_dict[key] = len(key_dict)

        box_key = 'Box%03d' % key_dict[key]

        if box_key in f.keys():
            grp = f[box_key]
        else:
            grp = f.create_group(box_key)

        if 'X' not in grp.keys():
            x_dset = grp.create_dataset('X', data=scatter.reshape((1, scatter.shape[0], scatter.shape[1], scatter.shape[2])),
                                        maxshape=(None, scatter.shape[0], scatter.shape[1], scatter.shape[2] ))
            y_dset = grp.create_dataset('Y', data=y.reshape((1, 2)), maxshape=(None, 2))

        else:
            x_dset, y_dset = grp['X'], grp['Y']

            l = len(x_dset)

            x_dset.resize((l + 1), axis=0)
            x_dset[-1] = scatter

            y_dset.resize((l + 1), axis=0)
            y_dset[-1] = y
