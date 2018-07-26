#!/bin/bash
"""
Convert the raw simulation data into a format the CNN can accept
"""
from os import path
from itertools import izip

import numpy as np
import pandas as pd
from glob import glob

"""
create a mock redshift survey given a mock with galaxy positions and velocities.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology
from astropy.constants import c  # the speed of light



def ra_dec_z(x, v=None, cosmo=None):
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
    
    if v is None:
        v = np.zeros_like(x)
    # calculate the observed redshift
    if cosmo is None:
        cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)
    c_km_s = c.to('km/s').value

    # remove h scaling from position so we can use the cosmo object
    x = x/cosmo.h

    # compute comoving distance from observer
    r = np.sqrt(x[:, 0]**2+x[:, 1]**2+x[:, 2]**2)

    # compute radial velocity
    ct = x[:, 2]/r
    st = np.sqrt(1.0 - ct**2)
    cp = x[:, 0]/np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    sp = x[:, 1]/np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    vr = v[:, 0]*st*cp + v[:, 1]*st*sp + v[:, 2]*ct

    # compute cosmological redshift and add contribution from perculiar velocity
    yy = np.arange(0, 1.0, 0.001)
    xx = cosmo.comoving_distance(yy).value
    f = interp1d(xx, yy, kind='cubic')
    z_cos = f(r)
    redshift = z_cos+(vr/c_km_s)*(1.0+z_cos)

    # calculate spherical coordinates
    theta = np.arccos(x[:, 2]/r)
    phi = np.arctan2(x[:, 1], x[:, 0])

    # convert spherical coordinates into ra,dec
    ra = phi
    dec = theta - np.pi/2.0

    return ra, dec, redshift

def convert_particles_to_density(directory,boxno, Lbox = 512, Lvoxel = 2, N_voxels_per_side = 4 ):

    reader = pd.read_csv(path.join(directory, 'uatu_z0p000.0'), delim_whitespace = True, chunksize = 5000)

    n_voxels = Lbox/Lvoxel
    particle_counts = np.zeros((n_voxels, n_voxels, n_voxels), dtype= int) # Lvoxel Mpc/h voxels

    for i, chunk in enumerate(reader):
        arr = chunk.values[:, :3]
        x,y,z = arr[:,0].astype(float), arr[:,1].astype(float), arr[:,2].astype(float)

        x_idx, y_idx, z_idx = np.floor_divide(x, Lvoxel).astype(int), np.floor_divide(y, Lvoxel).astype(int), np.floor_divide(z, Lvoxel).astype(int)

        for i,j,k in izip(x_idx, y_idx, z_idx):
            particle_counts[i,j,k]+=1
        #np.add.at(particle_counts, np.c_[x_idx, y_idx, z_idx], 1)

    # God has left this place
    # convert the histogram in a list of sub-voxels which will be hte input to the training set.
    x = np.array(np.split(particle_counts, N_voxels_per_side))
    y = np.vstack(np.split(x, N_voxels_per_side, axis = 2))
    voxel_list = np.vstack(np.split(y, N_voxels_per_side, axis = 3))

    print np.cbrt(np.sum(voxel_list))
    np.save(path.join(directory, 'particle_hist_%03d.npy'%boxno), voxel_list)

def convert_all_particles(directory, **kwargs):

    all_subdirs = glob(path.join(directory, 'Box*/'))
    for boxno, subdir in enumerate(sorted(all_subdirs)):
        print subdir
        convert_particles_to_density(subdir,boxno, **kwargs)

    # TODO delte the particles?

if __name__ == "__main__":
    from sys import argv

    directory = argv[1]
    convert_all_particles(directory)
