#!/bin/bash
#SBATCH --job-name=test_data_loading
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o test_data_loading.out 
#SBATCH --exclusive

python test_data_loading.py 
