#!/bin/bash
"""
This file computes initial power spectra and submits Picola simulations for training data.
"""

from os import path, mkdir
from subprocess import call
from time import time

import numpy as np
from classy import Class

from config_strings import picola_config


def omega_cdm_sample(lower = 0.1, upper = 0.15, N= 500):
    """
    Draw uniform samples of Omega_cdm from a flat prior.

    :param lower:
        Lower value of distribution. Default is 0.227, so O_m is 0.25
    :param upper:
        Upper value of distribution. Default is 0.327, so O_m is 0.3
    :param N:
        Number of samples to draw. Default is 500.
    :return:
        O_cdm, N uniformly sampled values of O_cdm
    """
    return np.random.uniform(lower, upper, N)

def A_s_sample(log_mean = 3.089, log_std = 0.036, N = 500):
    """
    Sample ln(10^10A_s), from the distribution used in Ravanbakhsh et al 2016.
    :param log_mean:
        The mean of ln(10^10 A_s), where we sample. Default is from Ravanbakhsh et al, and is Plank motivated.
    :param log_std:
        The std of ln(10^10 A_s), where we sample. Default is from Ravanbakhsh et al, and is Plank motivated.
    :param N:
        Number of samples to draw. Default is 500.
    :return:
        A_s, N normally sampled values of ln(10^10A_s)
    """

    return np.random.normal(log_mean,scale = log_std, size = N)

    #return np.exp(log_10_as)*1e-10

def make_sherlock_command(jobname, outputdir, \
                          picola_location = '/home/users/swmclau2/picola/l-picola/', max_time = 2):
    '''
    Return a list of strings that comprise a bash command to call picola on the cluster.
    Designed to work on sherlock's sbatch system. It must write a file
    to disk in order to work. Still returns a callable script.
    :param jobname:
        Name of the job. Will also be used to load the parameter file and make log files.
    :param max_time:
        Time for the job to run, in hours.
    :param outputdir:
        Directory to store output and param files.
    :return:
        Command, a string to call to submit the job.
    '''
    log_file = jobname + '.out'
    err_file = jobname + '.err'
    param_file = jobname + '.dat'
    sbatch_header = ['#!/bin/bash',
                     '--job-name=%s' % jobname,
                     '-p iric',  # KIPAC queue
                     '--output=%s' % path.join(outputdir, log_file),
                     '--error=%s' % path.join(outputdir, err_file),
                     '--time=%d:00:00' %max_time,  # max_time is in minutes
                     '--nodes=%d' % 1,
                     '--exclusive',
                     '--cpus-per-task=%d' % 16]

    sbatch_header = '\n#SBATCH '.join(sbatch_header)

    call_str = ['srun', path.join(picola_location, 'L-PICOLA'),
                path.join(outputdir, param_file)]

    call_str = ' '.join(call_str)
    # have to write to file in order to work.
    with open(path.join(outputdir, '%s.sbatch'%jobname), 'w') as f:
        f.write(sbatch_header + '\n' + call_str)

    return 'sbatch %s' % (path.join(outputdir, '%s.sbatch'%jobname))

def compute_pk(o_cdm, ln_10_As, outputdir):
    """
    Use class to compute the power spectrum and sigma 8 as initial conditions for the sims.
    :param o_cdm:
        Value of o_cdm to use in compuattion.
    :param ln_10_As:
        Vlaue of ln10^10A_s to use in computation
    :param outputdir:
        Outputdir to store the power specturm. It should be the same as where picola is loaded from.
    :return:
        sigma8. Return the value of sigma8 necessary to run the sims
    """

    z = 20.0 #Note maybe allow to vary, or make global
    params = {
        'output': 'mPk',
        'ln10^{10}A_s': ln_10_As,
        'P_k_max_h/Mpc': 100.0,
        'n_s': 0.96,
        'h': 0.7,
        'non linear': 'halofit',
        'omega_b': 0.022,
        'omega_cdm': o_cdm,
        'z_pk': z}

    cosmo = Class()
    cosmo.set(params)

    cosmo.compute()#level = ["initnonlinear"])

    k_size = 600
    ks = np.logspace(-3, 1.5, k_size).reshape(k_size,1,1)
    zs = np.array([z])

    pks =  cosmo.get_pk(ks, zs, k_size, 1, 1)[:,0,0]

    np.savetxt(path.join(outputdir, 'class_pk.dat'), np.c_[ks[:,0,0], pks],\
               delimiter = ' ')

    return cosmo.sigma8()

def write_picola_params(o_cdm, sigma_8, outputdir, jobname):

    fname = path.join(outputdir, '%s.dat'%jobname)

    #convert to actual Ocdm
    Ocdm = o_cdm/(0.7**2)
    Ob = 0.022/(0.7**2)
    Om = Ocdm+Ob


    formatted_config = picola_config.format(seed = time()%10000, #5001
                                            outputdir=outputdir,\
                                            file_base = jobname,
                                            ocdm = Ocdm,
                                            ob = Ob,
                                            olambda = 1 - Om,
                                            sigma8 = sigma_8)

    with open(fname, 'w') as f:
        f.write(formatted_config)

if __name__ == "__main__":
    from sys import argv
    N = int(argv[1])
    outputdir = argv[2] # any more args i'll use argparser
    jobname = 'uatu'

    o_cdm = omega_cdm_sample(N=N)
    ln10As = A_s_sample(N = N)
    Om = o_cdm/(0.7**2) + 0.022/(0.7**2)

    for idx, (o,om, a) in enumerate(zip(o_cdm,Om, ln10As)):
        sub_outputdir = path.join(outputdir, 'Box%03d'%idx)
        sub_jobname = jobname + '_%03d'%idx
        if not path.isdir(sub_outputdir):
            mkdir(sub_outputdir)

        with open(path.join(sub_outputdir, 'output_redshifts.dat'), 'w') as f:
            f.write("0.0, 10") #all we need

        sigma_8 = compute_pk(o, a, sub_outputdir)
        with open(path.join(sub_outputdir, 'input_params%03d.dat'%idx), 'w') as f:
            f.write("O_m: %f\nln10As: %f\nsigma_8: %f"%(om, a, sigma_8))

        write_picola_params(o, sigma_8, sub_outputdir, jobname)
        command = make_sherlock_command(jobname, sub_outputdir)

        call(command, shell=True)

