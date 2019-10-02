#!/bin/bash
#SBATCH --job-name=make_saliency_map
#SBATCH --time=1:00:00
#SBATCH --gres gpu:1
#SBATCH -p iric 
#SBATCH -o make_saliency_map.out 

python make_saliency_map.py 
