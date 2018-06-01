#!/bin/bash
#SBATCH --job-name=make_cov_mat
#SBATCH --time=4:00:00
#SBATCH -p iric 
#SBATCH -o make_cov_mat.out 
#SBATCH --exclusive

python make_cov_mat.py 
