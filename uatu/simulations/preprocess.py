#!/bin/bash
"""
Convert the raw simulation data into a format the CNN can accept
"""
from os import path

import numpy as np
import pandas as pd
from glob import glob

def convert_particles_to_density(directory, Lbox = 512, Lvoxel = 2, N_voxels_per_side = 64):

    reader = pd.read_csv(path.join(directory, 'uatu_z0p000.0'), sep = '\t', chunksize = 100)

    n_voxels = Lbox/Lvoxel
    particle_counts = np.zeros((n_voxels, n_voxels, n_voxels), dtype= int) # Lvoxel Mpc/h voxels

    for chunk in reader:
        x,y,z = chunk.data[:, :3]

        x_idx, y_idx, z_idx = np.mod(x, Lvoxel), np.mod(y, Lvoxel), np.mod(z, Lvoxel)

        particle_counts[x_idx, y_idx, z_idx]+=1

    # God has left this place
    # convert the histogram in a list of sub-voxels which will be hte input to the training set.
    x = np.array(np.split(particle_counts, N_voxels_per_side))
    y = np.vstack(np.split(x, N_voxels_per_side, axis = 2))
    voxel_list = np.vstrack(np.split(y, N_voxels_per_side, axis = 3))

    np.save(path.join(directory, 'particle_hist.npy'), voxel_list)

def convert_all_particles(directory, **kwargs):

    all_subdirs = glob(path.join(directory, 'Box*/'))

    for subdir in all_subdirs:
        convert_all_particles(subdir, **kwargs)

if __name__ == "__main__":
    from sys import argv

    dir = argv[0]
    convert_all_particles(dir)
