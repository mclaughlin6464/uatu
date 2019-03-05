from os import path
import numpy as np
from glob import glob
import h5py

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
            # TODO use the shape of the data
            a,b = np.random.randint(1, 3, size = 2) #randomly swap two axes, to rotate the input array
            # NOTE could flip, rotate axes as well
            return iter((np.swapaxes(self.X[i:i + B], a,b), self.Y[i:i + B]) for i in xrange(0, N, B))

class DatasetFromFile(object):
    def __init__(self, fname, batch_size, shuffle=False, augment = True, test_idxs = None,\
                 train_test_split=0.7, take_log = False):

        assert path.isfile(fname)

        self.batch_size, self.shuffle = batch_size, shuffle
        self.augment = augment
        
        self.fname = fname

        self.take_log = take_log

        f = h5py.File(fname, 'r')
        n_boxes = len(f.keys()) 

        start, stop = f.attrs['start'], f.attrs['stop']
        f.close()

        if test_idxs is None:
        
            if start is not None and stop is not None:
                box_idxs = np.arange(start, stop)
            else:
                box_idxs = np.arange(n_boxes) 

            # TODO magic numbers beware
            all_idxs = np.zeros((81*box_idxs.shape[0], 2))
            
            i = 0
            for bi in box_idxs: 
                for sbi in xrange(81):
                    all_idxs[i,0] = bi
                    all_idxs[i,1] = sbi
                    i+=1

            if shuffle:
                shuffled_idxs = np.arange(all_idxs.shape[0])
                np.random.shuffle(shuffled_idxs)
                all_idxs = all_idxs[shuffled_idxs]
            

            self.idxs = all_idxs[:int(all_idxs.shape[0]*train_test_split)]
            self.counter = 0

            if train_test_split != 1.0:
                self.test_idxs = all_idxs[int(all_idxs.shape[0]*train_test_split):]

        else: # test dset

            self.idxs = test_idxs
            self.counter = 0
            self.test_idxs = None

    def __iter__(self):

        N, B = len(self.idxs), self.batch_size
        return iter(self.__next__() for i in xrange(0, N, B)) 

    def __next__(self):
        
        f = h5py.File(self.fname, 'r')
        outputX, outputY = [] ,[]
        #print 'Next', self.counter, self.test_idxs is None
        for i in self.idxs[self.counter:self.counter+self.batch_size]:
            bn, sbn = i 
            X = f['Box%03d'%bn]['X'][sbn]
            Y = f['Box%03d'%bn]['Y'][sbn]
            if self.augment:
                a,b = np.random.randint(0, 2, size = 2) #randomly swap two axes, to rotate the input array
                X = np.swapaxes(X, a,b)
            if self.take_log:
                X = np.array(X).astype(float) # for some reason have to do this
                X[X<1e-3] = 1e-3
                X = np.log10(X)

            outputX.append(X)
            outputY.append(Y)
        f.close()

        self.counter=(self.counter + self.batch_size)%len(self.idxs)
        X = np.stack(outputX)
        Y = np.stack(outputY)
        return X,Y 


def get_xy_from_dir(dir, boxno):

    assert path.isdir(dir)

    #X = np.load(path.join(dir, 'particle_hist_%03d.npy'%boxno))
    X = np.load(path.join(dir, 'proj_map_%03d.npy'%boxno))
    #X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

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
            #assert X.shape[1] == 64
            Xs.append(X)
            Ys.append(Y)
        except IOError: #TODO only for testing!
            print 'Failed on %s'%subdir
        except ValueError: #wrong shape
            print 'Skipped %s'%subdir
        except AssertionError:
            print 'Skipped %s'%subdir

    return np.vstack(Xs), np.vstack(Ys)

def make_hdf5_file(dir,fname, start=None, stop = None):

    assert path.isdir(dir)
    all_subdirs = sorted(glob(path.join(dir, 'Box*/')))
    if stop is not None or start is not None:
        all_subdirs = all_subdirs[start:stop] 

    f = h5py.File(fname, 'w')
    f.attrs['start'] = start if start is not None else 0
    f.attrs['stop']  = stop if stop is not None else len(all_subdirs)

    x_mean = 0.0
    counter = 0
    basenames = []

    for boxno, subdir in enumerate(all_subdirs):
        bn = boxno+start if start is not None else boxno 
        print subdir
        X,Y = get_xy_from_dir(subdir, boxno)

        x_mean+=X.mean()
        counter+=1

        basename = subdir.split('/')[-2]
        basenames.append(basename)

        group = f.create_group(basename)
        group.create_dataset("X", data = X)
        group.create_dataset("Y", data = Y)

    f.attrs['shape'] = X.shape
    f.attrs['mean'] = x_mean/(counter*np.prod(X.shape))
    
    x_std = 0.0
    for boxno, basename in enumerate(basenames):
       x_std+=np.sum((f[basename]["X"].value()-f.attrs['mean'])**2) 
    
    f.attrs['std'] = x_std/(counter*np.prod(X.shape)) 

    f.close()
