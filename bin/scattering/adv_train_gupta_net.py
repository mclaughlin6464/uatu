from uatu.scattering import *
from uatu.watchers.Dataset import *
import torch
from os import path
from scipy.ndimage import gaussian_filter

shape = (256, 256)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

K = 1
model = FluriNet(K, p_dropout=0.0).to(device)

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuFastPMTraining/'
#dir = '/home/sean/Git/uatu/data/'
orig_fname = path.join(dir, 'UatuFastPMTraining.hdf5')

batch_size = 32  
smooth = int(argv[1]) 
noise = float(argv[2]) 
shape_noise = noise/np.sqrt((2.34**2)*30) #sigma_e/sqrt(A*n)
np.random.seed(0)
data_mod = lambda x: gaussian_filter(x+np.random.randn(*x.shape)*shape_noise, smooth) # add a normalization, hopefully sufficient
transform = torch.Tensor
#transform = lambda x : torch.Tensor(gaussian_filter(x, smoothing))
orig_train_dset = DatasetFromFile(orig_fname,batch_size, shuffle=True, augment=True, train_test_split = 0.7,\
                                 whiten = False, cache_size = 200, data_mod=data_mod, transform=transform)
orig_val_dset = orig_train_dset.get_test_dset()

#output_dir = '/home/sean/Git/uatu/networks/'
#print('Epoch 0')
#val_test(model, device, val_dset, scattering)
output_dir= '/home/users/swmclau2/scratch/uatu_networks/'
init_epoch = 0
if init_epoch>0:
    model_path = path.join(output_dir, 'fluri_net_reg_smooth_%0.1f_noise_%0.1f_%02d_adv.pth'%(smooth, noise,init_epoch))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Optimizer
lr = 5e-4
epochs = 10

for i in range(init_epoch):
    if i%5 and i>0:
        lr*=0.1

for epoch in range(init_epoch, epochs):
#if epoch%20==0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    if (epoch)%5==0 and epoch>0:
        lr*=0.1

#    if epoch > 0 and epoch%10==0:
#        lr*=0.5
#    if epoch> 3:
#        lr = 5e-7

    adv_train(model, device, orig_train_dset, optimizer, init_epoch+epoch+1, print_every=200)#, loss = 'mse')
    val_test(model, device, orig_val_dset)

    if epoch%1==0:
        torch.save(model.state_dict(), path.join(output_dir, 'fluri_net_reg_smooth_%0.1f_noise_%0.2f_epoch_%02d_adv.pth'%(smooth, noise,init_epoch+epoch)))


