from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path
from scipy.ndimage import gaussian_filter

#downsample_factor = 4
dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTest/'
#dir = '/home/sean/Git/uatu/data/'
orig_fname = path.join(dir, 'UatuFastPMTest.hdf5')
#clone_fname  = path.join(dir, 'UatuLightconeTrainingRobustifyDeepResnetAdvGRF.hdf5')


batch_size = 8  

smooth = 0#1
noise = 0.0#0.29#29
shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)

epoch = 9#10 
np.random.seed(0)

data_mod = lambda x: np.log10(gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth)+1.0) # add a normalization, hopefully sufficient
transform = torch.Tensor
#train_dset = DatasetFromFile(clone_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
#                                 whiten = False, cache_size = 100, transform=transform)
#val_dset = train_dset.get_test_dset()

orig_test_dset = DatasetFromFile(orig_fname,batch_size, shuffle=False, augment=False, train_test_split =1.0,\
                                 whiten = False, cache_size = 200, data_mod=data_mod, transform=transform)

output_dir= '/home/users/swmclau2/scratch/uatu_preds/'
output_fname = path.join(output_dir, 'adv_gupta_net_smooth_%0.1f_noise_%0.1f_%02d.hdf5'%(smooth, noise,epoch))

shape = (256, 256)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
K = 1
model = GuptaNet(K, p_dropout=0.0).to(device)

model_path = '/home/users/swmclau2/scratch/uatu_networks/gupta_net_reg_smooth_%0.1f_noise_%0.2f_epoch_%02d_adv.pth'%(smooth, noise,epoch)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

test(model, device, orig_test_dset, output_fname)
