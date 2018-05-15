#!/bin/bash
#SBATCH --job-name=submit_jobs
#SBATCH --time=2:00:00
#SBATCH -p iric 
#SBATCH -o submit_jobs 
#SBATCH --ntasks=1

python uatu/simulations/submit_sims.py 500 /home/users/swmclau2/scratch/UatuTraining/
