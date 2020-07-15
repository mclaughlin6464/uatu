#!/bin/bash
#SBATCH --job-name=scattering_mcmc
#SBATCH --time=4:00:00
###SBATCH --array=0-50
#SBATCH -p iric,kipac
#SBATCH --ntasks=16
###SBATCH --exclusive
#SBATCH -o scattering_mcmc.out 
###SBATCH -o peak_counts_mcmc_23.out 

python scattering_mcmc.py #$SLURM_ARRAY_TASK_ID
