#!/bin/bash
"""
Convert the raw simulation data into a format the CNN can accept
"""
from os import path
from itertools import izip, product
from collections import OrderedDict

import numpy as np
import pandas as pd
from glob import glob
#from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.interpolate import interp1d
from astropy import cosmology
from astropy.constants import c  # the speed of light
from numba import jit

def ra_dec_z(x, cosmo=None):
    """
    Lifted from halotools, originally written by Duncan Campbell

    Calculate the ra, dec, and redshift assuming an observer placed at (0,0,0).

    Parameters
    ----------
    x: array_like
        Npts x 3 numpy array containing 3-d positions in Mpc/h

    v: array_like
        Npts x 3 numpy array containing 3-d velocities in km/s

    cosmo : object, optional
        Instance of an Astropy `~astropy.cosmology` object.  The default is
        FlatLambdaCDM(H0=0.7, Om0=0.3)

    Returns
    -------
    ra : np.array
        right accession in radians

    dec : np.array
        declination in radians

    redshift : np.array
        "observed" redshift

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    We do the same thing to assign random peculiar velocities:

    >>> vx,vy,vz = (np.random.random(Npts),np.random.random(Npts),np.random.random(Npts))
    >>> vels = np.vstack((vx,vy,vz)).T

    >>> from astropy.cosmology import WMAP9 as cosmo
    >>> ra, dec, redshift = ra_dec_z(coords, vels, cosmo = cosmo)
    """
    if len(x.shape) == 1:
        x = np.array([x])
    
    # calculate the observed redshift
    if cosmo is None:
        cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)

    # remove h scaling from position so we can use the cosmo object
    x = x/cosmo.h

    # compute comoving distance from observer
    r = np.sqrt(x[:, 0]**2+x[:, 1]**2+x[:, 2]**2)

    # compute cosmological redshift and add contribution from perculiar velocity
    yy = np.arange(0, 1.0, 0.001)
    xx = cosmo.comoving_distance(yy).value
    f = interp1d(xx, yy, kind='cubic')
    redshift = f(r)

    # calculate spherical coordinates
    theta = np.arccos(x[:, 2]/r)
    phi = np.arctan2(x[:, 1], x[:, 0])
    ra = phi
    dec = theta - np.pi/2.0

    return ra, dec, redshift

def get_astropy_cosmo(directory, boxno):
    """
    Get the astropy cosmology according to the parameters of this box
    """
    with open(path.join(directory, 'input_params%03d.dat'%boxno), 'r') as f:
        for line in f:
            if line[0] == 'O':
                splitline = line.split(':')
                omega_m = float(splitline[-1])

            elif line[0] == 's':
                splitline = line.split(':')
                sigma_8 = float(splitline[-1])

                # don't use sigma_8, not sure it matters
    return cosmology.FlatLambdaCDM(H0 = 70, Om0 = omega_m, Ob0 = 0.022/(0.7**2) )

def default_bias_model(delta, params):
    return delta*params['b1']+params['b2']*delta**2 

def make_LHC(ordered_params, N):
    
    np.random.seed(int(time()))

    points = []
                                                                                                                                        # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for plow, phigh in ordered_params.itervalues():
        point = np.linspace(plow, phigh, num=N)
        np.random.shuffle(point)  # makes the cube random.
        points.append(point)
    
    return np.stack(points).T

def apply_bias_model(box, n_points = 100, ordered_params = None, bias_model = default_bias_model):

    if ordered_params is None:
        if bias_model == default_bias_model:
            ordered_params = OrderedDict({'b1':(0, 2), 'b2':(-1, 1)})
        else:
            raise ValueError("Please specified ordered params")

    lhc = make_LHC(ordered_params, n_points)

    bias_models = np.zeros((n_points. box.shape[0], box.shape[0], box.shape[0])) 

    rho_bar = box.mean()
    delta = (box-rho_bar)/(rho_bar) 

    for idx, point in enumerate(lhc): 
        params = dict(zip(ordered_params.keys(), point))
        bias_models[idx] = bias_model(delta, params)

    return lhc, bias_models

@jit
def convert_box_to_proj_density(directory, boxno,box, Lbox = 512.0, ang_size_image = 2, pixels_per_side = 32, n_z_bins = 4):
    # ang size image: size of each sub image in degrees
    # pixels_per_side: number of pixels per side in the sub images

    cosmo = get_astropy_cosmo(directory, boxno) 
    #establish min/max bounds

    test_coords = np.c_[np.array([0.0, 0.0, 0.0, 0.0, 512.0, 512.0, 512.0, 512.0]),\
                        np.array([0.0, 0.0, 512.0, 512.0, 0.0, 0.0, 512.0, 512.0]),\
                        np.array([0.0, 512.0, 0.0, 512.0, 0.0, 512.0, 0.0, 512.0])] 

    ra,dec,z = ra_dec_z(test_coords, cosmo=cosmo)
    ra, dec = np.degrees(ra), np.degrees(dec)
    min_ra, max_ra = np.nanmin(ra), np.nanmax(ra)
    min_dec, max_dec = np.nanmin(dec), np.nanmax(dec)
    min_z, max_z = 0.0, np.nanmax(z)

    z_bin_size = max_z/n_z_bins

    # assuming ra and dec spacing the same
    n_pixels = int( (max_ra - min_ra)*pixels_per_side/ang_size_image )
    #unsolved: how to determine max resfhit? 
    proj_density = np.zeros((n_pixels, n_pixels, n_z_bins))

    idxs = np.array(list(product(range(box.shape[0]), repeat=3)))
    Lvoxel = Lbox/box.shape[0]
    coords = Lvoxel*(idxs.astype(float)+0.5)

    ra, dec, z = ra_dec_z(coords, cosmo=cosmo)
    ra, dec = np.degrees(ra), np.degrees(dec)

    ra_idx, dec_idx = np.floor_divide(ra-min_ra, ang_size_image*1.0/pixels_per_side).astype(int), np.floor_divide(dec-min_dec, ang_size_image*1.0/pixels_per_side).astype(int)
    z_idx = np.floor_divide(z, z_bin_size).astype(int)

    ra_idx[ra_idx<0] = 0
    ra_idx[ra_idx >=n_pixels] = n_pixels-1 #edge cases

    dec_idx[dec_idx<0] = 0
    dec_idx[dec_idx >= n_pixels] = n_pixels-1 #edge cases

    for ri, di, zi, density  in izip(ra_idx, dec_idx, z_idx, np.nditer(box)):
        proj_density[ri, di, zi] += density #would like to avoid this for loop, hopefully numba helps

    # God has left this place
    # convert the histogram in a list of sub-voxels which will be hte input to the training set.
    x = np.array(np.split(proj_density, (max_ra - min_ra)/ang_size_image))
    pixel_list = np.vstack(np.split(x, (max_dec-min_dec)/ang_size_image, axis = 2))

    return pixel_list
    #np.save(path.join(directory, 'particle_hist_%03d.npy'%boxno), pixel_list)

# TODO clarify syntax between this and above? work a little differency
@jit
def convert_particles_to_density(directory,boxno, Lbox = 512, Lvoxel = 2, N_voxels_per_side = 4, full = True):
    # Lvoxel, size on an individual density voxel
    # number of voxels in a subvolume
    if path.isfile(path.join(directory, 'uatu_lightcone.0' )): # is a lightcone
        reader = pd.read_csv(path.join(directory, 'uatu_lightcone.0'), delim_whitespace = True, chunksize = 5000)
    else:
        reader = pd.read_csv(path.join(directory, 'uatu_z0p000.0'), delim_whitespace = True, chunksize = 5000)

    n_voxels = Lbox/Lvoxel
    particle_counts = np.zeros((n_voxels, n_voxels, n_voxels), dtype= int) # Lvoxel Mpc/h voxels
    for i, chunk in enumerate(reader):
        arr = chunk.values[:, :3]
        x,y,z = arr[:,0].astype(float), arr[:,1].astype(float), arr[:,2].astype(float)
        x_idx, y_idx, z_idx = np.floor_divide(x, Lvoxel).astype(int), np.floor_divide(y, Lvoxel).astype(int), np.floor_divide(z, Lvoxel).astype(int)
        x_idx[x_idx <0] = 0
        x_idx[x_idx >= particle_counts.shape[0]] = particle_counts.shape[0]-1
        y_idx[y_idx <0] = 0
        y_idx[y_idx >= particle_counts.shape[0]] = particle_counts.shape[0]-1
        z_idx[z_idx <0] = 0
        z_idx[z_idx >= particle_counts.shape[0]] = particle_counts.shape[0]-1
        for i,j,k in izip(x_idx, y_idx, z_idx):
            particle_counts[i,j,k]+=1
            #count_in_bins(particle_counts, x_idx, y_idx, z_idx)
            #np.add.at(particle_counts, np.c_[x_idx, y_idx, z_idx], 1)
            # God has left this place
            # convert the histogram in a list of sub-voxels which will be hte input to the training set.
    x = np.array(np.split(particle_counts, N_voxels_per_side))
    y = np.vstack(np.split(x, N_voxels_per_side, axis = 2))
    voxel_list = np.vstack(np.split(y, N_voxels_per_side, axis = 3))

    np.save(path.join(directory, 'particle_hist_%03d.npy'%boxno), voxel_list)

def convert_all_particles(directory, **kwargs):

    all_subdirs = glob(path.join(directory, 'Box*/'))
    for boxno, subdir in enumerate(sorted(all_subdirs)):
        print subdir
        #if path.isfile(path.join(subdir, 'uatu_lightcone.info' )): # is a lightcone
        #    convert_particles_to_proj_density(subdir, boxno, **kwargs)
        #else:
        convert_particles_to_density(subdir,boxno, **kwargs)

    # TODO delte the particles?

if __name__ == "__main__":
    from sys import argv

    directory = argv[1]
    convert_all_particles(directory)
