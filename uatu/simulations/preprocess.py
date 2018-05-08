#!/bin/bash
"""
Convert the raw simulation data into a format the CNN can accept
"""
from os import path

import numpy as np
import pandas as pd

def convert_particles_to_density(directory, Lbox = 512, Lvoxel = 2):

    reader = pd.read_csv(path.join(directory, 'uatu_z0p000.0'), sep = '\t', chunksize = 100)

    n_voxels = Lbox/Lvoxel
    particle_counts = np.zeros((n_voxels, n_voxels, n_voxels), dtype= int) # Lvoxel Mpc/h voxels

    for chunk in reader:
        x,y,z = chunk.data[:, :3]

        x_idx, y_idx, z_idx = np.mod(x, Lvoxel), np.mod(y, Lvoxel), np.mod(z, Lvoxel)

        particle_counts[x_idx, y_idx, z_idx]+=1

    np.save(path.join(directory, 'particle_hist.npy'))

