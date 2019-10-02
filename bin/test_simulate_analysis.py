from glob import glob
import numpy as np
from nbodykit.source.catalog import CSVCatalog
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit.cosmology.correlation import CorrelationFunction
from nbodykit.cosmology.power.halofit import HalofitPower
from nbodykit.cosmology import Cosmology
from os import path

def compute_xi_from_box(bins, boxdir):
    names  = ['x', 'y','z','vx','vy','vz']
    f = CSVCatalog(path.join(boxdir, 'uatu_z0p000.0'), names)

    downsample_idxs = np.random.choice(f['x'].shape[0], int(1e-3*f['x'].shape[0]), replace = False)
    d_idxs = np.zeros((f['x'].shape[0],))
    d_idxs[downsample_idxs] = 1
    f = f[d_idxs.astype(bool)]
    f['Position'] = np.c_[f['x'], f['y'], f['z']]
    s = SimulationBox2PCF('1d', f, bins, BoxSize = 512.0, show_progress = True)

    return np.array([x[0] for x in s.corr.data])

bins = np.loadtxt('/home/users/swmclau2/scratch/uatu_bins.npy')
print bins

boxdir = '/home/users/swmclau2/scratch/UatuTraining4/Box000/'

xi = compute_xi_from_box(bins, boxdir)
print xi

from sys import exit
exit(0)

for Om in np.linspace(0.25, 0.35, 5):
    for s8 in np.linspace(0.7, 0.9, 5):
        h = 1.0
        Ocdm = Om*(h**2) - 0.022

        cosmo = Cosmology(h=h, T0_cmb=2.726,
                              Omega_b = 0.022/(h**2),
                              Omega0_cdm=Ocdm,
                              P_k_max = 100.0,
                                n_s=0.96, nonlinear = True).match(s8)
        bc = (bins[1:] + bins[:-1])/2.0
        predicted_xi = CorrelationFunction(HalofitPower(cosmo, 0.0))(bc)

        print Om, s8
        print cosmo.Omega0_m, cosmo.sigma8
        print predicted_xi
        print np.sum((xi-predicted_xi)**2)
        print '*'*25
