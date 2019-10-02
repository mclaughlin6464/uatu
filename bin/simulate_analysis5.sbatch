#!/bin/bash
#SBATCH --job-name=simulate_analysis5
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o simulate_analysis5.out 
#SBATCH --exclusive

module load openmpi/2.1.1
python simulate_analysis5.py 
