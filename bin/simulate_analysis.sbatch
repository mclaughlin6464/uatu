#!/bin/bash
#SBATCH --job-name=simulate_analysis
#SBATCH --time=24:00:00
#SBATCH -p iric 
#SBATCH -o simulate_analysis.out 
#SBATCH --exclusive

python simulate_analysis.py 
