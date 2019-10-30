from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from kymatio import Scattering2D
from os import path
from time import time

mode = 0 # 0, 1, or 2
max_order = 2 if mode == 2 else 1
J = 2 # 0, 1, or
shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

scattering = Scattering2D(J=J, shape=shape, max_order=max_order)

L = 8
#K = 1 + L*J+ (L**2)*(J*(J-1))/2.0

if mode == 0:
    # do a little fuckery
    scattering = lambda x: scattering(x)[:,0] # TODO double check this
    K = 1
if mode == 1:
    K = 1 + L*J
elif mode == 2:
    K = 1 + L*J +(L**2)*(J*(J-1))/2.0

if use_cuda:
    scattering = scattering.cuda()


width = 2

model = Scattering2dResNet(K, width).to(device)


t0 = time()
dir = '/scratch/users/swmclau2/UatuLightconeTraining/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

print(path.isdir(fname) )


batch_size = 32


train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8, whiten = True, cache_size = 50, transform=torch.Tensor)
val_dset = train_dset.get_test_dset()

data = (train_dset, val_dset, None)

# Optimizer
lr = 1e-4
epochs = 15

output_dir= '/home/users/swmclau2/scratch/uatu_networks/'

save_path = path.join(output_dir, 'scattering_resnet.pth')

for epoch in range(0, epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #lr*=0.2

    train(model, device, train_dset, optimizer, epoch+1, scattering)
    val_test(model, device, val_dset, scattering)

    torch.save(model.state_dict(), save_path)


# TODO cacheing?
# TODO custom loss funciton
