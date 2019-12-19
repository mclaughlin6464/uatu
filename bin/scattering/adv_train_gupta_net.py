from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from kymatio import Scattering2D
from os import path
from time import time
from sys import argv
from scipy.ndimage import gaussian_filter

J = 0 # 0, 1, or
shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

K = 1

model = GuptaNet(K).to(device)

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
#dir = '/home/sean/Git/uatu/data/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

batch_size = 32  
smoothing = 0

transform = lambda x : torch.Tensor(gaussian_filter(x, smoothing))

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
                                 whiten = True, cache_size = 100, transform=transform)
val_dset = train_dset.get_test_dset()

data = (train_dset, val_dset, None)

# Optimizer
lr = 1e-4
epochs = 30 

output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
#output_dir = '/home/sean/Git/uatu/networks/'

for epoch in range(epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #lr*=0.2

    adv_train(model, device, train_dset, optimizer, epoch+1)
    val_test(model, device, val_dset, scattering)

    if epoch%1==0:
        torch.save(model.state_dict(), path.join(output_dir, 'adv_gupta_net_smooth_%d_epoch_%02d.pth'%(smoothing, epoch)))


