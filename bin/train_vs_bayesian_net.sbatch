#!/bin/bash
#SBATCH --job-name=train_vs_bayesian_net
#SBATCH --time=24:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o train_vs_bayesian_net.out 

python train_vs_bayesian_net.py 
