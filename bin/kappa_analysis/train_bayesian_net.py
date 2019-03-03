from uatu.watchers import *
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from os import path
from time import time
t0 = time()
dir = '/scratch/users/swmclau2/UatuLightconeTraining/'
#fname = path.join(dir, 'data.hdf5')

def whiten(X):
    mu = X.mean()
    s = X.std()
    return (X-mu)/s

X,Y = get_all_xy(dir)#, 64, shuffle=True, augment=True)
X = whiten(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size = 0.8, shuffle = False)
#X_val, X_test, y_val, y_test = train_test_split(X,Y, train_size = 0.9, shuffle = False)
batch_size = 32 
train_dset = Dataset(X_train, y_train, batch_size, shuffle = True, augment = True)
test_dset = Dataset(X_test, y_test, batch_size, shuffle = True, augment = True)

#val_dset = Dataset(X_val,y_val, 30, shuffle=True, augment=True)
#test_dset = DatasetFromFile(fname, 64, shuffle=True, augment=True, test_idxs = train_dset.test_idxs)

data = (train_dset, test_dset, None)

#best vals
#lr 1e-4
#epochs 15
#lam 1e-9
#dropout 0.5
# standard cost
train(gupta_bayesian_network_init_fn, standard_optimizer_init_fn, bayes_cost_fn,\
        data, num_epochs = 20, \
        fname = '/home/users/swmclau2/scratch/gupta_bayesian_net_kappa', print_every = 500,\
         lr_np = 1e-4, lam_np = 0.0, rate_np = 0.5, bayes_prob = 0.9) 
