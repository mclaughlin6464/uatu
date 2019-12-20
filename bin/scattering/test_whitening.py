import numpy as np
from uatu.watchers.Dataset import *
from os import path
from sys import argv
from scipy.ndimage import gaussian_filter

dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
#dir = '/home/sean/Git/uatu/data/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')


#transform = lambda x : torch.Tensor(gaussian_filter(x, smoothing))

batch_size = 1#32 
train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8,\
                                 whiten = True, cache_size = 10)#, transform=transform)

for batch_idx, (data, target) in enumerate(train_dset):
    print( batch_idx,  data.mean(), data.std())#, target)
