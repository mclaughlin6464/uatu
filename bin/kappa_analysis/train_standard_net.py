from uatu.watchers import *
from sklearn.model_selection import train_test_split
from os import path
from time import time
t0 = time()
dir = '/scratch/users/swmclau2/UatuLightconeTraining/'
#fname = path.join(dir, 'data.hdf5')

X,Y = get_all_xy(dir)#, 64, shuffle=True, augment=True)
X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size = 0.8, shuffle = True)
train_dset = Dataset(X_train, y_train, 30, shuffle = True, augment = True)
test_dset = Dataset(X_test, y_test, 30, shuffle = True, augment = True)

#val_dset = Dataset(X_val,y_val, 30, shuffle=True, augment=True)
#test_dset = DatasetFromFile(fname, 64, shuffle=True, augment=True, test_idxs = train_dset.test_idxs)

data = (train_dset, test_dset, None)

device = "/device:GPU:0"
#device = '/cpu:0'
print time() - t0
train(standard_convnet_init_fn, standard_optimizer_init_fn, standard_cost_fn, data, device, num_epochs = 100, fname = '/home/users/swmclau2/scratch/standard_net_kappa', print_every = 500)#, lr = 0.0001) 
