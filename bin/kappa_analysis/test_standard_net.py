from uatu.watchers import *
from sklearn.model_selection import train_test_split
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

_, X_test, _, y_test = train_test_split(X,Y, train_size = 0.8, shuffle = False)
#X_val, X_test, y_val, y_test = train_test_split(X,Y, train_size = 0.9, shuffle = False)
batch_size = 1 #currently required by my testing code, chagne if slow?
test_dset = Dataset(X_test, y_test, batch_size, shuffle = False, augment = False)

test(gupta_network_init_fn, test_dset, n_samples = 1,\
     fname = '/home/users/swmclau2/scratch/gupta_net_kappa',
     samples_fname='/home/users/swmclau2/scratch/gupta_net_kappa_samples.hdf5')
