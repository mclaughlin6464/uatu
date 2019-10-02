#!/bin/bash
#SBATCH --job-name=test_standard_net
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_standard_net.out 

python test_standard_net.py 
