from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path
from scipy.ndimage import gaussian_filter
from sys import argv

#downsample_factor = 4
shape = (256, 256)
in_channels = 1
width = 2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
depth = 3#[16, 32, 64, 64, 64, 64, 64]
model = DeepResnet(input_size = shape[0], in_channels=in_channels, n_subplanes=width, depth=depth).to(device)


dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
dir2 = '/scratch/users/swmclau2/clone_maps/'
#dir = '/home/sean/Git/uatu/data/'
smooth = int(argv[1]) 
noise = float(argv[2]) 
if smooth==0:
    orig_fname = path.join(dir, 'UatuFastPMTraining.hdf5')
else:
    orig_fname = path.join(dir, 'UatuFastPMTraining_smooth_1.0_noise_0.3.hdf5')
#np.random.seed(0)
clone_fname  = path.join(dir2, 'UatuFastPMTrainingRobustifyDeepResnetRegAdvWhiteNoise%0.1f_v8.hdf5'%smooth)
#clone_fname  = path.join(dir2, 'UatuFastPMTrainingRobustifyDeepResnetRegAdvGRFNoise%0.1f_v3.hdf5'%smooth)
                             #UatuFastPMTrainingRobustifyDeepResnetRegAdvGRFNoise1.hdf5
batch_size = 32#16  
A = 11.8#hp.nside2pixarea(1024, True)*(60**2)
shape_noise = noise/np.sqrt(A*30) #sigma_e/sqrt(A*n)
np.random.seed(0)
#data_mod = lambda x: x # add a normalization, hopefully sufficient
data_mod = lambda x: x#gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth) # add a normalization, hopefully sufficient
transform = torch.Tensor
train_dset = DatasetFromFile(clone_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.7,\
                                 whiten = True, cache_size = 200, data_mod=lambda x:x, transform=transform)
val_dset = train_dset.get_test_dset()

orig_train_dset = DatasetFromFile(orig_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.7,\
                                 whiten = True,whiten_vals=(train_dset.mean, train_dset.std), cache_size = 200, data_mod=data_mod, transform=transform)
orig_val_dset = orig_train_dset.get_test_dset()

#data = (train_dset, val_dset, None)

output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
init_epoch = 0
output_fname = 'deep_resnet_reg_smooth_%0.1f_noise_%0.1f_%02d_white_clone_v8.pth'
if init_epoch>0:
    model_path = path.join(output_dir, output_fname%(smooth, noise,init_epoch)) 
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Optimizer
lr = 1e-4
epochs = 10

for i in range(init_epoch):
    if i%5 and i>0:
        lr*=0.5
print(output_fname)
for epoch in range(init_epoch, epochs):
    #if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
    if (epoch)%5==0 and epoch>0:
        lr*=0.5

    train(model, device, train_dset, optimizer, epoch+1)#,attack_lr = 1e-2, attack_nsteps = 5)#, smoothing = 1)
    #adv_train(model, device, train_dset, optimizer, epoch+1, attack_lr = 1e-2, attack_nsteps = 5,alpha=0.0)#, smo4
    val_test(model, device, val_dset)
    val_test(model, device, orig_val_dset)

    if epoch%1==0:
        torch.save(model.state_dict(), path.join(output_dir, output_fname%(smooth, noise,epoch)))

