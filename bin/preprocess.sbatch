#!/bin/bash
#SBATCH --job-name=preprocess2
#SBATCH --time=48:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/uatu/preprocess2.out 
#SBATCH --ntasks=16

python ~swmclau2/Git/uatu/uatu/simulations/preprocess.py /home/users/swmclau2/scratch/UatuTest4/
