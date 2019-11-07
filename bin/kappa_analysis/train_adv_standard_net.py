from uatu.watchers import *
from os import path
from time import time
t0 = time()
dir = '/oak/stanford/orgs/kipac/users/swmclau2/Uatu/UatuLightconeTraining/'
#dir = '/home/sean/Git/uatu/data/'
#fname = path.join(dir, 'UatuLightconeTraining.hdf5')
fname = path.join(dir, 'UatuLightconeAttackedShuffled_v2.hdf5')

batch_size = 32 

train_dset = DatasetFromFile(fname,batch_size, shuffle=True, augment=True, train_test_split = 0.8, whiten = True, cache_size = 50, y_key = 'target_Y')
test_dset = train_dset.get_test_dset() 
test_dset.y_key = 'true_Y'

data = (train_dset, test_dset, None)

#best vals
#lr 1e-4
#epochs 15
#lam 1e-6
#dropout 0.2
# standard cost
train(gupta_adv_network_init_fn, standard_optimizer_init_fn, adversarial_standard_abs_cost_fn, data, num_epochs = 100, fname = '/scratch/users/swmclau2/uatu_networks/gupta_net_kappa_adv_abs_shuffled_attack_v2', print_every = 1000, lr_np = 1e-4, lam_np = 1e-6, adv = True)
