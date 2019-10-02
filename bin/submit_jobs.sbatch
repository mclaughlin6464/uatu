#!/bin/bash
#SBATCH --job-name=submit_jobs
#SBATCH --time=2:00:00
#SBATCH -p iric 
#SBATCH -o submit_jobs.out
#SBATCH --ntasks=1

#python ~swmclau2/Git/uatu/uatu/simulations/submit_sims.py 250 /home/users/swmclau2/scratch/UatuLightconeTraining/

module load openmpi/2.1.1
module load fftw

python ~swmclau2/Git/uatu/uatu/simulations/submit_sims.py 5 /home/users/swmclau2/scratch/UatuLightconeSameSeed/ 5001
