#!/bin/bash
#SBATCH --job-name=train_standard_net
#SBATCH --time=02:00:00
#SBATCH --gres gpu:4
#SBATCH -p gpu 
#SBATCH --ntasks=12
#SBATCH -o train_standard_net.out 

module unload py-jupyter
module load py-scipystack 
module load py-tensorflow
#module load openmpi/2.1.1
module load fftw
module load gsl
#module load intelmpi/4.1.3.048 intel/13sp1.2.144
module load python/2.7.13

python train_standard_net.py 
