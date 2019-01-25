#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=01:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/uatu/bin/preprocess.out 
#SBATCH --exclusive

python ~swmclau2/Git/uatu/uatu/simulations/preprocess.py /home/users/swmclau2/scratch/UatuLightconeTraining/
