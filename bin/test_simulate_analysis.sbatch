#!/bin/bash
#SBATCH --job-name=test_simulate_analysis
#SBATCH --time=12:00:00
#SBATCH -p iric 
#SBATCH -o test_simulate_analysis.out 
#SBATCH --exclusive

module load openmpi/2.1.1
python test_simulate_analysis.py 
