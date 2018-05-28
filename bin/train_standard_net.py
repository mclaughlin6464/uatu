from uatu.watchers import *
from sklearn.model_selection import train_test_split
from os import path

dir = '/scratch/users/swmclau2/UatuTraining2/'
#dir = '/home/users/swmclau2/scratch/UatuTraining/'

#X, y = get_all_xy(dir)
X, y = get_xy_from_dir( path.join(dir, 'Box000'), 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

#X_val,X_test, y_val, t_test = train_test_split(X_test, y_test, train_size = 0.6)

train_dset = Dataset(X_train,y_train, 30, shuffle=True, augment=True)
#val_dset = Dataset(X_val,y_val, 30, shuffle=True, augment=True)
test_dset = Dataset(X_test, y_test, 30, shuffle=True, augment=True)

data = (train_dset, test_dset, None)

device = "/device:GPU:0"
#device = '/cpu:0'

train(standard_convnet_init_fn, standard_optimizer_init_fn, data, device, num_epochs = 100) 
