from os import path
from uatu.simulations.preprocess import *
boxno = 0
directory = "/home/users/swmclau2/scratch/UatuLightconeTraining/Box%03d/"%boxno 

print 0
#convert_particles_to_density(directory, boxno, full = True)
print 1
full_box = np.load(path.join(directory, 'full_particle_hist_%03d.npy'%boxno)) 

proj_box_list , full_proj_box = convert_box_to_proj_density(directory, boxno, box = full_box, pixels_per_side = 16,  n_z_bins = 1)
print 2

#np.save(path.join(directory, 'proj_particle_hist_%03d.npy'%boxno), proj_box_list)
np.save(path.join(directory, 'full_proj_box_%03d.npy'%boxno), full_proj_box)

proj_box_list = np.load(path.join(directory, 'proj_particle_hist_%03d.npy'%boxno))
lhc, bias_boxes = apply_bias_model(full_box)
print 3

np.save(path.join(directory, 'bias_lhc_%03d.npy'%boxno), lhc)

# this hsould be some kinda helper function in preprocess
print 4
for i, bb in enumerate(bias_boxes):
    print i
    proj_bb, full_bb = convert_box_to_proj_density(directory, boxno, box=bb, pixels_per_side = 16, n_z_bins = 1)
    np.save(path.join(directory, 'proj_bias_hist_idx_%03d_box_%03d.npy'%(i, boxno)), proj_bb)
    np.save(path.join(directory, 'full_bias_hist_idx_%03d_box_%03d.npy'%(i, boxno)), full_bb)
    break
