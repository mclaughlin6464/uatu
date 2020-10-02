# Just scatter a whole dataset, so we can do PCA, etc, on them.
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func
from astropy.units import deg
from lenstools import GaussianNoiseGenerator, ConvergenceMap
from scipy.ndimage import gaussian_filter

shape = (256, 256)
smooth = 1
dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
fname = path.join(dir, 'UatuFastPMTraining.hdf5')
smooth = 0
noise = 0.0#29

output_fname  = path.join(dir, 'UatuFastPMTrainingGRF_smooth_%0.1f_noise_%0.1f.hdf5'%(smooth, noise))

batch_size =32 
shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)
np.random.seed(0)
data_mod = lambda x: gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth) # add a normalization, hopefully sufficient
attrs = {}
with h5py.File(fname, 'r') as f:
    for key in f.attrs.keys():
        attrs[key] = f.attrs[key]

train_dset = DatasetFromFile(fname,batch_size, shuffle=False, augment=False,
                             train_test_split = 1.0, whiten = False, cache_size = 64, data_mod = data_mod)

key_dict = {}

ls = np.linspace(128, 128**2, 180)
side_angle = 10*deg
gen = GaussianNoiseGenerator(shape=shape,side_angle=side_angle)
x0_shape = (batch_size, shape[0], shape[1])
l = int(len(train_dset)*1.0/batch_size)
with h5py.File(path.join(dir, output_fname), 'w') as f:
    for key, val in attrs.items():
        f.attrs[key] = val

    for i, (xt,y) in enumerate(train_dset):
        xt = xt.squeeze()
        #x0 = torch.Tensor(np.random.randn(xt.shape[0], shape[0], shape[1]))
        # TODO repeat this with smoothing?
        x0 = np.zeros_like(xt)
        for i,_xt in enumerate(xt):
            cmap = ConvergenceMap(gaussian_filter(_xt, smooth), angle=side_angle)
            l1, psd1D = cmap.powerSpectrum(ls)
            gaussian_map = gen.fromConvPower(np.array([l1,psd1D]),seed=1,kind="linear",bounds_error=False,fill_value=0.0)
            _x0 = gaussian_map.data

            x0[i] = ((_x0-_x0.mean())/_x0.std())*_xt.std()+_xt.mean() 
        #print(x0.shape, xt.shape)
        robust_x = x0  
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
                x_dset = grp.create_dataset('X', data=robust_x[i].reshape((n_y, robust_x.shape[1], robust_x.shape[2])),
                                            maxshape=(None, robust_x.shape[1], robust_x.shape[2], ))
                y_dset = grp.create_dataset('Y', data=np.tile(_y, (n_y,1)), maxshape=(None, 2))

            else:
                x_dset, y_dset = grp['X'], grp['Y']

                l = len(x_dset)

                x_dset.resize((l + n_y ), axis=0)
                x_dset[-n_y:] = robust_x[i] 

                y_dset.resize((l + n_y), axis=0)
                y_dset[-n_y:] = _y
