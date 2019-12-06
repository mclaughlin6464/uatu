# Just scatter a whole dataset, so we can do PCA, etc, on them.
from kymatio import Scattering2D
import torch
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func
from uatu.scattering import Scattering2dResNet
from uatu.scattering import compute_robust_map 

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

 model_path = '/home/sean/Git/uatu/networks/scattering_resnet_max_mode_2_J_2_adv_20.pth'
 model = Scattering2dResNet(K, J, k=width).to(device)
 model.load_state_dict(torch.load(model_path, map_location='cpu'))
 model.eval()

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

output_fname  = path.join(dir, 'UatuLightconeTrainingRobustified.hdf5')

batch_size =32 
train_dset = DatasetFromFile(fname,batch_size, shuffle=False, augment=False,
                             train_test_split = 1.0, whiten = True, cache_size = 64, transform=torch.Tensor)

key_dict = {}

np.random.seed(64)

x0_shape = (batch_size, shape[0], shape[1])

with h5py.File(path.join(dir, output_fname), 'w') as f:
    for i, (xt,y) in enumerate(train_dset):
        print(i, y)
        x0 = torch.Tensor(np.random.randn(*x0_shape))
        x0 = x0*xt.std()+xt.mean()
        robust_x = compute_robust_map(scattering, model, x0.to(device), xt.to(device)).cpu().numpy()

        unique_ys, inv_idxs = np.unique(y.reshape((batch_size, 2)), return_inverse = True)
        y_idxs =  [np.where(inv_idxs == i)[0] for i in xrange(len(unique_ys))]  

        for  _y, i  in zip(unique_ys, y_idxs) 
            n_y = _y.shape[0]
            key = key_func(_y[0].reshape((1, 2)))
            if key not in key_dict:
                key_dict[key] = len(key_dict)
        
            box_key = 'Box%03d' % key_dict[key]

            if box_key in f.keys():
                grp = f[box_key]
            else:
                grp = f.create_group(box_key)

            if 'X' not in grp.keys():
                x_dset = grp.create_dataset('X', data=robust_x[i].reshape((n_y, robust_x.shape[1], robust_x.shape[2], robust_x.shape[3])),
                                            maxshape=(None, robust_x.shape[1], robust_x.shape[2], robust_x.shape[3] ))
                y_dset = grp.create_dataset('Y', data=_y.reshape((n_y, 2)), maxshape=(None, 2))

            else:
                x_dset, y_dset = grp['X'], grp['Y']

                l = len(x_dset)

                x_dset.resize((l + n_y ), axis=0)
                x_dset[-n_y:] = robust_x[i] 

                y_dset.resize((l + n_y), axis=0)
                y_dset[-n_y:] = _y
