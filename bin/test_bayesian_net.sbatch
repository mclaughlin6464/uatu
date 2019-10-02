#!/bin/bash
#SBATCH --job-name=test_bayesian_net
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_bayesian_net.out 

python test_bayesian_net.py 
