# Just scatter a whole dataset, so we can do PCA, etc, on them.
import torch
from os import path
import h5py
from uatu.watchers.Dataset import *
from uatu.watchers.test import key_func
from uatu.scattering import * 
from astropy.units import deg


shape = (256, 256)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

depth = 5

smooth =1 
noise = 0.29#0.00
epoch = 9#10 
#model_path = '/scratch/users/swmclau2/uatu_networks/deep_resnet_50_adv.pth'
output_dir = '/scratch/users/swmclau2/uatu_networks/'
model_path = path.join(output_dir, 'deep_resnet_reg_smooth_%0.1f_noise_%0.1f_%02d_adv2.pth'%(smooth, noise,epoch))
model = DeepResnet(input_size=shape[0], in_channels=in_channels, n_subplanes=width, depth=depth).to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
fname = path.join(dir, 'UatuFastPMTraining.hdf5')

output_fname  = path.join(dir, 'UatuFastPMTrainingRobustifyDeepResnetRegAdvWhiteNoise%0.1f.hdf5'%smooth) 
#output_fname  = path.join(dir, 'UatuFastPMTrainingRobustifyDeepResnetRegAdvGRFNoise%0.1f.hdf5'%smooth)
#grf_fname = path.join(dir, 'UatuFastPMTrainingGRF_smooth_%0.1f_noise_%0.1f.hdf5'%(smooth, noise))
shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)
np.random.seed(0)
data_mod = lambda x: np.log10(gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth)+1.0) # add a normalization, hopefully sufficient
transform = torch.Tensor

last_key = None
last_idx = None

batch_size = 16#8

key_dict = None 
with h5py.File(fname, 'r') as f1, h5py.File(output_fname) as f2:
    for key in f1.attrs.keys():
        if key not in f2.attrs:
            f2.attrs[key] = f1.attrs[key] 

    if len(f2.keys())>0:
        last_key = list(f2.keys())[-1]
        last_idx = len(f2[last_key]['X'])

    if 'key_dict' in f2.attrs.keys():
        key_dict = eval(f2.attrs['key_dict'])

if key_dict is None:
    key_dict = {}

print("Last Key", last_key)
if last_key is not None:
    start_idx = (int(last_key[-3:])*1296+last_idx)/batch_size
else:
    start_idx = 0

print('Start Idx: %d'%start_idx)

train_dset = DatasetFromFile(fname,batch_size, shuffle=False, augment=False,
                             train_test_split = 1.0, whiten = False, cache_size = 128, data_mod=data_mod, transform=transform)

#grf_dset = DatasetFromFile(grf_fname,batch_size, shuffle=False, augment=False,
#                             train_test_split = 1.0, whiten = False, cache_size = 64, data_mod = data_mod, transform=transform)


scattering = lambda x:x
np.random.seed(64)
x0_shape = (batch_size, shape[0], shape[1])
l = int(len(train_dset)*1.0/batch_size)
print('Beginning')
#for i, ((xt,y), (x0,_)) in enumerate(zip(train_dset, grf_dset)):
for i, (xt,y) in enumerate(train_dset):

# TODO start batches
    if i< start_idx:
        print('Skipping %d'%i, flush=True)
        continue
    print(i, flush=True)
    xt = xt.squeeze()
    x0 = torch.Tensor(np.random.randn(*xt.shape))*xt.std()+xt.mean()
    x0 = x0.to(device)#.squeeze()
    #x0 = torch.Tensor(np.random.randn(xt.shape[0], shape[0], shape[1]))
    #x0 =  ((x0-x0.mean())/x0.std())*xt.std()+xt.mean() 

    robust_x = compute_robust_map(scattering, device, model, x0, xt).cpu().detach().numpy()

    unique_ys, first_idxs, inv_idxs = np.unique(y.reshape((xt.shape[0], 2))[:,0],return_index=True, return_inverse = True)#, axis=0)
    
    y_idxs =  [np.where(inv_idxs == i)[0] for i in range(len(unique_ys))]  

    unique_ys = y[first_idxs, :]
    if len(unique_ys.shape)==1:
        np.expand_dims(unique_ys, axis=0)

    with h5py.File(path.join(dir, output_fname)) as f:
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

        f.attrs['key_dict'] = str(key_dict)
