from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path
from sys import argv

#downsample_factor = 4
shape = (256, 256)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
depth = 3#[16, 32, 64, 64, 64, 64, 64]
model = DeepResnet(input_size = shape[0], in_channels=in_channels, n_subplanes=width, depth=depth, shuffle_layers=1).to(device)

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
#dir = '/home/sean/Git/uatu/data/'
orig_fname = path.join(dir, 'UatuFastPMTraining.hdf5')
#clone_fname  = path.join(dir, 'UatuLightconeTrainingRobustifyDeepResnetAdvGRF.hdf5')


batch_size = 8 
smooth =  int(argv[1]) 
noise = float(argv[2]) 

shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)
np.random.seed(0)
data_mod = lambda x: gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth) # add a normalization, hopefully sufficient
transform = torch.Tensor

orig_train_dset = DatasetFromFile(orig_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.7,\
                                 whiten = False, cache_size = 200, data_mod=data_mod, transform=transform)
orig_val_dset = orig_train_dset.get_test_dset()

data = (orig_train_dset, orig_val_dset, None)

# Optimizer
output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
init_epoch = 0
if init_epoch>0:
    model_path = path.join(output_dir, 'deep_resnet_shuffle_reg_smooth_%0.1f_noise_%0.1f_%02d_v6.pth'%(smooth, noise,init_epoch))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Optimizer
lr = 5e-4
epochs = 10 

for i in range(init_epoch):
    if i%3 and i>0:
        lr*=0.2
        #pass

for epoch in range(init_epoch, epochs):
#if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if (epoch)%3==0 and epoch>0:
        lr*=0.2
        #pass

    train(model, device, orig_train_dset, optimizer, epoch+1)#, smoothing = 1)
    val_test(model, device, orig_val_dset)

    if epoch%1==0:
        torch.save(model.state_dict(), path.join(output_dir, 'deep_resnet_shuffle_reg_smooth_%0.1f_noise_%0.1f_%02d_v3.pth'%(smooth, noise, epoch))) 

