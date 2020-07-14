#!/bin/bash
#SBATCH --job-name=smc
#SBATCH --time=4:00:00
#SBATCH -p iric,kipac
#SBATCH --ntasks=8
###SBATCH --exclusive
#SBATCH -o scattering_mcmc.sbatch 


python scattering_mcmc.sbatch
