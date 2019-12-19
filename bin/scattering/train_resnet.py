from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from kymatio import Scattering2D
from os import path
from time import time
from sys import argv

mode = 0
J = 0 # 0, 1, or
shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

scattering = lambda x:x 
K = 1
width = 2

model = Scattering2dResNet(K, J, k=width).to(device)


t0 = time()
#dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
dir = '/home/sean/Git/uatu/data/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

batch_size = 2 

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.3, whiten = True, cache_size = 10, transform=torch.Tensor)
val_dset = train_dset.get_test_dset()

data = (train_dset, val_dset, None)

# Optimizer
lr = 1e-4
epochs = 30 

#output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
output_dir = '/home/sean/Git/uatu/networks/'

for epoch in range(epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #lr*=0.2

    train(model, device, train_dset, optimizer, epoch+1, scattering, print_every=500)
    val_test(model, device, val_dset, scattering)

    #torch.save(model.state_dict(), save_path)

    if epoch%1==0:
        torch.save(model.state_dict(), path.join(output_dir, 'resnet_max_mode_%d_J_%d_epoch_%02d.pth'%(mode, J, epoch)))


