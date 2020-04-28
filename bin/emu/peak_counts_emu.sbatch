#!/bin/bash
#SBATCH --job-name=pce
#SBATCH --time=4:00:00
#SBATCH -p iric,kipac
#SBATCH --ntasks=8
###SBATCH --exclusive
#SBATCH -o peak_counts_emu.out 


python peak_counts_emu.py
