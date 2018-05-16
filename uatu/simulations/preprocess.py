#!/bin/bash
"""
Convert the raw simulation data into a format the CNN can accept
"""
from os import path
from itertools import izip

import numpy as np
import pandas as pd
from glob import glob

def convert_particles_to_density(directory, Lbox = 512, Lvoxel = 2, N_voxels_per_side = 4):

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
    np.save(path.join(directory, 'particle_hist.npy'), voxel_list)

def convert_all_particles(directory, **kwargs):

    all_subdirs = glob(path.join(directory, 'Box*/'))
    for subdir in sorted(all_subdirs):
        print subdir
        convert_particles_to_density(subdir, **kwargs)

    # TODO delte the particles?

if __name__ == "__main__":
    from sys import argv

    directory = argv[1]
    convert_all_particles(directory)
