from uatu.watchers.train import * 
from uatu.watchers.Dataset import *
from uatu.watchers.networks import *
from sklearn.model_selection import train_test_split
from os import path
from time import time
t0 = time()
#dir = '/scratch/users/swmclau2/UatuTraining2/'
dir = '/home/users/swmclau2/scratch/UatuTraining4/'
fname = path.join(dir, 'data.hdf5')


train_dset = DatasetFromFile(fname, 64, shuffle=True, augment=True)
#val_dset = Dataset(X_val,y_val, 30, shuffle=True, augment=True)
test_dset = DatasetFromFile(fname, 64, shuffle=True, augment=True, test_idxs = train_dset.test_idxs)

data = (train_dset, test_dset, None)

device = "/device:GPU:0"
#device = '/cpu:0'
print time() - t0
train(shallow_bayesian_convnet_init_fn, standard_optimizer_init_fn, bayes_cost_fn, data, device, num_epochs = 100, fname = '/home/users/swmclau2/scratch/UatuCheckpoints/bayes_shallow2', print_every = 500)
