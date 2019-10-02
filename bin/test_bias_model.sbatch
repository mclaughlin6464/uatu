#!/bin/bash
#SBATCH --job-name=preprocess_000
#SBATCH --time=01:00:00
#SBATCH -p iric 
#SBATCH -o /home/users/swmclau2/Git/uatu/bin/test_bias_model.out 
#SBATCH --ntasks=16
###SBATCH --exclusive

python /home/users/swmclau2/Git/uatu/bin/test_bias_model.py
