from uatu.watchers import *
from sklearn.model_selection import train_test_split
from time import time

dir = '/scratch/users/swmclau2/UatuLightconeTest/'
#fname = path.join(dir, 'UatuLightconeTest.hdf5')
#fname = path.join(dir, 'UatuLightconeAttackedBayesRenorm.hdf5')
fname = path.join(dir, 'UatuLightconeAttackedBayesTranspose.hdf5')

test_dset = DatasetFromFile(fname, batch_size=1, shuffle = False, augment = False,
                            whiten=True, train_test_split=1.0, cache_size = 1)

test(gupta_bayesian_network_init_fn, test_dset, n_samples = 100,\
     fname = '/scratch/users/swmclau2/uatu_networks/gupta_bayes_net_kappa-45000',
     #samples_fname=path.join(dir, 'UatuLightconeBayesPreds.hdf5'))
     #samples_fname=path.join(dir, 'UatuLightconeAttackedRenormBayesPreds.hdf5'))
     samples_fname=path.join(dir, 'UatuLightconeAttackedBayesTranposePreds.hdf5'))
