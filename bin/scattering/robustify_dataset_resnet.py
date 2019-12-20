# Just scatter a whole dataset, so we can do PCA, etc, on them.
from kymatio import Scattering2D
import torch
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func
from uatu.scattering import Scattering2dResNet
from uatu.scattering import compute_robust_map 
from sys import argv

mode = 0
J = 0 # 0, 1, or
shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

scattering = lambda x: x 
K =1
width = 2

model_path = argv[1] 
model = Scattering2dResNet(K, J, k=width).to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

#dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
dir = '/home/sean/Git/uatu/data/'

fname = path.join(dir, 'UatuLightconeTraining.hdf5')


output_fname  = path.join(dir, argv[2]) 

batch_size = 8
train_dset = DatasetFromFile(fname,batch_size, shuffle=False, augment=False,
                             train_test_split = 1.0, whiten = True, cache_size = 100, transform=torch.Tensor)

key_dict = {}

np.random.seed(64)

x0_shape = (batch_size, shape[0], shape[1])

with h5py.File(path.join(dir, output_fname), 'w') as f:
    for i, (xt,y) in enumerate(train_dset):
        print(i, y[0])
        x0 = torch.Tensor(np.random.randn(*x0_shape))
        x0 = x0*xt.std()+xt.mean()
        robust_x = compute_robust_map(scattering, device, model, x0, xt).cpu().detach().numpy()

        unique_ys,first_idxs, inv_idxs = np.unique(y.reshape((batch_size, 2))[:,0], return_inverse = True, return_index = True)
        y_idxs =  [np.where(inv_idxs == i)[0] for i in range(len(unique_ys))]  
        unique_ys = np.array(y).reshape((batch_size, 2))[first_idxs,:]
        for  _y, i  in zip(unique_ys, y_idxs):
            n_y = i.shape[0]# if len(_y.shape)>0 else 1
            key = key_func(np.array(_y).reshape((1, 2)))
            if key not in key_dict:
                key_dict[key] = len(key_dict)
        
            box_key = 'Box%03d' % key_dict[key]

            if box_key in f.keys():
                grp = f[box_key]
            else:
                grp = f.create_group(box_key)

            if 'X' not in grp.keys():
                x_dset = grp.create_dataset('X', data=robust_x[i].reshape((n_y, robust_x.shape[1], robust_x.shape[2])),
                                            maxshape=(None, robust_x.shape[1], robust_x.shape[2]))

                y_dset = grp.create_dataset('Y', data=np.tile(_y, (n_y, 1)), maxshape=(None, 2))

            else:
                x_dset, y_dset = grp['X'], grp['Y']

                l = len(x_dset)

                x_dset.resize((l + n_y ), axis=0)
                x_dset[-n_y:] = robust_x[i] 

                y_dset.resize((l + n_y), axis=0)
                y_dset[-n_y:] = np.tile(_y, (n_y,1))
