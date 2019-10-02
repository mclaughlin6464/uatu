from uatu.watchers import *
from os import path
from time import time
t0 = time()
dir = '/scratch/users/swmclau2/UatuLightconeTraining/'
fname = path.join(dir, 'UatuLightconeTraining.hdf5')

print path.isdir(fname)


#X,Y = get_all_xy(dir)#, 64, shuffle=True, augment=True)
#X = whiten(X)

#X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size = 0.8, shuffle = False)
#X_val, X_test, y_val, y_test = train_test_split(X,Y, train_size = 0.9, shuffle = False)
batch_size = 32 

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8, whiten = True, cache_size = 50)
test_dset = train_dset.get_test_dset() 

data = (train_dset, test_dset, None)

#best vals
#lr 1e-4
#epochs 15
#lam 1e-6
#dropout 0.2
# standard cost
train(gupta_network_init_fn, standard_optimizer_init_fn, adversarial_standard_abs_cost_fn, data, num_epochs = 50, fname = '/home/users/swmclau2/scratch/uatu_networks/gupta_net_kappa_adv_abs', print_every = 100, lr_np = 2e-4, lam_np = 0.0)
