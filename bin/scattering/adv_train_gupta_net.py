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

model = GuptaNet(K, p_dropout=0.0).to(device)

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
#dir = '/home/sean/Git/uatu/data/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

batch_size = 32  
smoothing = 0

#transform = lambda x : torch.Tensor(gaussian_filter(x, smoothing))
transform = torch.Tensor

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
                                 whiten = True, cache_size = 100, transform=transform)
val_dset = train_dset.get_test_dset()

data = (train_dset, val_dset, None)

scattering = lambda x: x
# Optimizer
lr = 2e-6
epochs = 50 

output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
#output_dir = '/home/sean/Git/uatu/networks/'
#print('Epoch 0')
#val_test(model, device, val_dset, scattering)

for epoch in range(epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    if epoch > 0 and epoch%10==0:
        lr*=0.9
#    if epoch> 3:
#        lr = 5e-7

    # doing a test where i've changed the adv loss to be the same as a normal loss
    # let's see if that leads to similar training
    adv_train(model, device, train_dset, optimizer, epoch+1, print_every=200, loss = 'mae', attack_lr=1e-6)
    val_test(model, device, val_dset, scattering)

    if epoch%5==0:
        torch.save(model.state_dict(), path.join(output_dir, 'gupta_net_smooth_%d_epoch_%02d_adv_test3.pth'%(smoothing, epoch)))


