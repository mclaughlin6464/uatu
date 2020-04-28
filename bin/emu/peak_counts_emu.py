import numpy as np 
import h5py
from lenstools import ConvergenceMap
from scipy.ndimage import gaussian_filter
from astropy.units import deg


# In[ ]:


training_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTraining/UatuFastPMTraining.hdf5'
test_filename = '/home/users/swmclau2/oak/Uatu/UatuFastPMTest/UatuFastPMTest.hdf5'


# In[ ]:


with h5py.File(training_filename, 'r') as f:
    mean, std = f.attrs['mean'], f.attrs['std']
    X =  f['Box000']['X'][()].squeeze()


shape_noise = 0.3/np.sqrt((2.34**2)*30)

def noise_and_smooth(image, noise_level = shape_noise, smooth=1):
    i = image + np.random.randn(*image.shape)*noise_level
    return gaussian_filter(i, smooth)


# In[ ]:


smoothX = noise_and_smooth(X[0])



def image_pc(image):
    #image = (image-image.mean())/image.std()
    cmap = ConvergenceMap(image, angle=10*deg)
    thresholds = np.linspace(-0.01, 0.04,  41)
    #thresholds = np.linspace(-5, 5, 200)
    nu,peaks = cmap.peakCount(thresholds, norm=False)
    return nu,peaks#/psd1D[0]


# In[ ]:


nu,pc = image_pc(smoothX)


def compute_all_pc(images):
    
    all_pcs = np.zeros((images.shape[0], 40))
    for i, im in enumerate(images):
        nu, all_pcs[i] = image_pc(noise_and_smooth(im))
    return nu, all_pcs


# In[ ]:


nu, all_pcs = compute_all_pc(X)


# In[ ]:




# In[ ]:


# create the peak counts datasets

def create_peak_counts_dset(conv_dset_fname):
        
    with h5py.File(conv_dset_fname, 'r') as cf:
        
        mean_pc = np.zeros((len(cf.keys()), 40))
        #err_pc =  np.zeros((len(cf.keys()), 40))
        cov_pc = np.zeros((len(cf.keys()), 40, 40))
        for i,key in enumerate(cf.keys()):
            print key
            X =  cf[key]['X'][()].squeeze()
            Y =  cf[key]['Y'][()].squeeze()
            
            _, all_pcs = compute_all_pc(X)
            
            mean_pc[i] = all_pcs.mean(axis=0)
            #err_pc[i] = all_pcs.std(axis=0)
            cov_pc[i] = np.cov(all_pcs, rowvar=False)#.cov(axis=0)
            
        return mean_pc, cov_pc

training_pc, training_cov = create_peak_counts_dset(training_filename)
# In[ ]:


test_pc, test_cov = create_peak_counts_dset(test_filename)

np.save('training_pc.npy', training_pc)
np.save('training_cov.npy', training_cov)
# In[ ]:


np.save('test_pc.npy', test_pc)
np.save('test_cov.npy', test_cov)



