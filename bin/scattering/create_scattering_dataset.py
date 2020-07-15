# Just scatter a whole dataset, so we can do PCA, etc, on them.
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func
from scipy.ndimage import gaussian_filter
import torch
from kymatio import Scattering2D

shape = (256, 256)
#dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
#fname = path.join(dir, 'UatuFastPMTraining.hdf5')

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTest/'
fname = path.join(dir, 'UatuFastPMTest.hdf5')

smooth = 0
noise = 0.0#29
np.random.seed(0)
data_mod = lambda x: gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth)#+1.0) # add a normalization, hopefully sufficient
device = 'cuda'
transform = lambda x: torch.Tensor(x).to(device)

shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)
output_fname  = path.join(dir, 'UatuFastPMTestScattering_smooth_%0.1f_noise_%0.1f.hdf5'%(smooth, noise))

batch_size = 16 
attrs = {}
with h5py.File(fname, 'r') as f:
    for key in f.attrs.keys():
        attrs[key] = f.attrs[key]

train_dset = DatasetFromFile(fname,batch_size, shuffle=False, augment=False,
                             train_test_split = 1.0, whiten = False, cache_size = 64,\
                              data_mod = data_mod, transform=transform)

J = 4
L = 4
shape = (256,256)

ls0 =1
ls1 = L*J
ls2 = L*L*J*J#(J-1)#/2

scattering = Scattering2D(J=J, shape=shape, max_order=2, L = L).cuda()

def get_scattering(m):
    S = scattering(m).mean((2,3), keepdim=True)#.to('cpu').numpy()

    s0 = S[:, :ls0].to(device).squeeze().unsqueeze(1)
    s1 = S[:, ls0:ls0+ls1].reshape((-1, J, L)).mean((2,), keepdim=True).squeeze()#.unsqueeze(1)
    s2 = S[:, ls0+ls1:].reshape((-1, J,L,J,L)).mean((2,4), keepdim=True).squeeze().reshape(\
                             (s0.shape[0], J*J))#.unsqueeze(1)
    #print(s0.shape, s1.shape, s2.shape)
    return torch.cat([s0, s1, s2], dim = 1)

key_dict = {}

with h5py.File(path.join(dir, output_fname), 'w') as f:
    for key, val in attrs.items():
        f.attrs[key] = val

    for i, (xt,y) in enumerate(train_dset):
        print(i)
        xt = xt.squeeze()
        #x0 = torch.Tensor(np.random.randn(xt.shape[0], shape[0], shape[1]))
        # TODO repeat this with smoothing?
        st = get_scattering(xt).to('cpu')
        y = y.cpu()
        unique_ys, first_idxs, inv_idxs = np.unique(y.reshape((xt.shape[0], 2))[:,0],return_index=True, return_inverse = True)#, axis=0)
        
        y_idxs =  [np.where(inv_idxs == i)[0] for i in range(len(unique_ys))]  

        unique_ys = y[first_idxs, :]
        if len(unique_ys.shape)==1:
            np.expand_dims(unique_ys, axis=0)

        for  _y, i  in zip(unique_ys, y_idxs):

            n_y = i.shape[0]
            key = key_func(np.expand_dims(_y, axis = 0))
            if key not in key_dict:
                key_dict[key] = len(key_dict)
        
            box_key = 'Box%03d' % key_dict[key]

            if box_key in f.keys():
                grp = f[box_key]
            else:
                grp = f.create_group(box_key)

            if 'X' not in grp.keys():
                x_dset = grp.create_dataset('X', data=st[i].reshape((n_y,int(1+J+J*J) )),
                                            maxshape=(None, int(1+J+J*J) ))
                y_dset = grp.create_dataset('Y', data=np.tile(_y, (n_y,1)), maxshape=(None, 2))

            else:
                x_dset, y_dset = grp['X'], grp['Y']

                l = len(x_dset)

                x_dset.resize((l + n_y ), axis=0)
                x_dset[-n_y:] = st[i] 

                y_dset.resize((l + n_y), axis=0)
                y_dset[-n_y:] = _y
