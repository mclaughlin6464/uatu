from os import path
import numpy as np
from glob import glob

class Dataset(object):
    def __init__(self, X, Y, batch_size, shuffle=False, augment = True):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == Y.shape[0], 'Got different numbers of data and labels'
        self.X, self.Y = X, Y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.augment = augment

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        if not self.augment:
            return iter((self.X[i:i + B], self.Y[i:i + B]) for i in xrange(0, N, B))
        else: #augment randomly on the fly
            a,b = np.random.randint(1, 4, size = 2) #randomly swap two axes, to rotate the input array
            # NOTE could flip, rotate axes as well
            return iter((np.swapaxes(self.X[i:i + B], a,b), self.Y[i:i + B]) for i in xrange(0, N, B))

def get_xy_from_dir(dir, boxno):

    assert path.isdir(dir)

    X = np.load(path.join(dir, 'particle_hist_%03d.npy'%boxno))
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
    with open(path.join(dir, 'input_params%03d.dat'%boxno)) as f:
        for line in f:
            if line[0] == 'O':
                splitline = line.split(':')
                omega_m = float(splitline[-1])# + 0.022 # TODO FIX BARYONS

            elif line[0] == 's':
                splitline = line.split(':')
                sigma_8 = float(splitline[-1])

    Y = np.zeros((X.shape[0],2))
    Y[:,0] = omega_m
    Y[:,1] = sigma_8

    return X, Y

def get_all_xy(dir, max = None):

    assert path.isdir(dir)
    Xs, Ys = [], []
    all_subdirs = sorted(glob(path.join(dir, 'Box*/')))
    if max is not None:
        all_subdirs = all_subdirs[:max] 

    for boxno, subdir in enumerate(all_subdirs):
        print subdir
        try:
            X,Y = get_xy_from_dir(subdir, boxno)
            assert X.shape[1] == 64
            Xs.append(X)
            Ys.append(Y)
        except IOError: #TODO only for testing!
            print 'Failed on %s'%subdir
        except ValueError: #wrong shape
            print 'Skipped %s'%subdir
        except AssertionError:
            print 'Skipped %s'%subdir

    return np.vstack(Xs), np.vstack(Ys)



