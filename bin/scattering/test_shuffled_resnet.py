from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path
from scipy.ndimage import gaussian_filter
from sys import argv
#downsample_factor = 4
dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTest/'
#dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'

#dir = '/home/sean/Git/uatu/data/'
orig_fname = path.join(dir, 'UatuFastPMTest.hdf5')
#orig_fname = path.join(dir, 'UatuFastPMTraining.hdf5')

#clone_fname  = path.join(dir, 'UatuLightconeTrainingRobustifyDeepResnetAdvGRF.hdf5')


batch_size = 8  

smooth = int(argv[1]) 
noise = float(argv[2]) 
shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)

epoch = 9
np.random.seed(0)

data_mod = lambda x: gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth) # add a normalization, hopefully sufficient
transform = torch.Tensor
#train_dset = DatasetFromFile(clone_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
#                                 whiten = False, cache_size = 100, transform=transform)
#val_dset = train_dset.get_test_dset()

orig_test_dset = DatasetFromFile(orig_fname,batch_size, shuffle=False, augment=False, train_test_split =1.0,\
                                 whiten = False, cache_size = 200, data_mod=data_mod, transform=transform)

output_dir= '/home/users/swmclau2/scratch/uatu_preds/'
output_fname = path.join(output_dir, 'deep_resnet_shuffle_reg_smooth_%0.1f_noise_%0.1f_%02d_v6.hdf5'%(smooth, noise,epoch))
#deep_resnet_shuffle2_reg_smooth_%0.1f_noise_%0.1f_%02d.pth

shape = (256, 256)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
depth = 3#[16, 32, 64, 64, 64, 64, 64]
model = DeepResnet(input_size = shape[0], in_channels=in_channels, n_subplanes=width, depth=depth,
                   shuffle_layers=1).to(device)

model_path = '/home/users/swmclau2/scratch/uatu_networks/deep_resnet_shuffle_reg_smooth_%0.1f_noise_%0.1f_%02d_v3.pth'%(smooth, noise,epoch)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

test(model, device, orig_test_dset, output_fname)
