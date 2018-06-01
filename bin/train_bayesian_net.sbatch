#!/bin/bash
#SBATCH --job-name=train_bayesian_net
#SBATCH --time=8:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o train_bayesian_net.out 
#SBATCH --exclusive

python train_bayesian_net.py 
