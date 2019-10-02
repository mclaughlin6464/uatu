#!/bin/bash
#SBATCH --job-name=simulate_analysis4
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o simulate_analysis4.out 
#SBATCH --exclusive

module load openmpi/2.1.1
python simulate_analysis4.py 
