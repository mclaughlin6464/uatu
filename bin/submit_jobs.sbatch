#!/bin/bash
#SBATCH --job-name=submit_jobs
#SBATCH --time=2:00:00
#SBATCH -p iric 
#SBATCH -o submit_jobs.out
#SBATCH --ntasks=1

python ~swmclau2/Git/uatu/uatu/simulations/submit_sims.py 250 /home/users/swmclau2/scratch/UatuTraining4/
