from uatu.watchers import *
from sklearn.model_selection import train_test_split
from os import path
from time import time
t0 = time()
#dir = '/scratch/users/swmclau2/UatuTraining2/'
dir = '/home/users/swmclau2/scratch/UatuTest4/'
#dir = '../data/'
fname = path.join(dir, 'data.hdf5')

data = DatasetFromFile(fname, 64, shuffle=False, augment=False,train_test_split = 1.0)

device = "/device:GPU:0"
#device = '/cpu:0'
test(very_shallow_convnet_init_fn, data,1, device,'/home/users/swmclau2/scratch/vs_standard_net-2000', '/home/users/swmclau2/scratch/vs_standard_net_preds') 
