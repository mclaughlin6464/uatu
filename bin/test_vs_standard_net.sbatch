#!/bin/bash
#SBATCH --job-name=test_vs_standard_net
#SBATCH --time=1:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_vs_standard_net.out 

python test_vs_standard_net.py 
