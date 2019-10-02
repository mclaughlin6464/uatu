#!/bin/bash
#SBATCH --job-name=train_standard_net
#SBATCH --time=00:20:00
#SBATCH --gres gpu:1
#SBATCH -p iric
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH -o train_standard_net.out 

module unload py-jupyter
module load py-scipystack 
#module load openmpi/2.1.1
module load fftw
module load gsl
#module load intelmpi/4.1.3.048 intel/13sp1.2.144
module load python/2.7.13
module load py-tensorflow/1.9.0_py27

python train_adv_standard_net.py 
