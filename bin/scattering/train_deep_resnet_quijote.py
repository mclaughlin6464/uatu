from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path

#downsample_factor = 4
shape = (128, 128)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
depth = 5#[16, 32, 64, 64, 64, 64, 64]
model = DeepResnet(input_size = shape[0],init_downsample_factor = 2, in_channels=in_channels, n_subplanes=width, depth=depth).to(device)


fname = '/scratch/users/swmclau2/UatuQuijote/UatuQuijoteTraining.hdf5'

batch_size = 16  

transform = torch.Tensor

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
                                 whiten = True, cache_size = 100, transform=transform)
val_dset = train_dset.get_test_dset()

data = (train_dset, val_dset, None)

# Optimizer
lr = 1e-4
epochs = 100 

output_dir= '/home/users/swmclau2/scratch/uatu_networks/'

for epoch in range(epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if epoch%25==0 and epoch>0:
        lr*=0.1

    train(model, device, train_dset, optimizer, epoch+1)#, smoothing = 1)
    val_test(model, device, val_dset)

    if epoch%10==0:
        torch.save(model.state_dict(), path.join(output_dir, 'deep_resnet_quijote_%02d.pth'%(epoch)))

