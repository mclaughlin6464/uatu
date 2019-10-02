#!/bin/bash
#SBATCH --job-name=test_standard_net
#SBATCH --time=01:00:00
#SBATCH --gres gpu:4
#SBATCH -p gpu, iric
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH -o /home/users/swmclau2/Git/uatu/bin/kappa_analysis/test_standard_net.out 

module unload py-jupyter
module load py-scipystack 
module load py-tensorflow
#module load openmpi/2.1.1
module load fftw
module load gsl
#module load intelmpi/4.1.3.048 intel/13sp1.2.144
module load python/2.7.13

python test_standard_net.py 
