"""
This file will host global variables of config strings for the sims.

We will only be varying one or two parameters, but we will need to write several copies of the all the parameters we
have to specify. I believe this is the easiest way to do that.
"""

# NOTE with the class python wrapper this one is not necessary.
class_config = """
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*  CLASS input parameter file  *
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

> This example of input file does not include all CLASS possibilities, but only the most useful ones for running in the basic LambdaCDM framework. For all possibilities, look at the longer 'explanatory.ini' file.
> You can use an even more concise version, in which only the arguments in which you are interested would appear.
> Only lines containing an equal sign not preceded by a sharp sign "#" are considered by the code. Hence, do not write an equal sign within a comment, the whole line would be interpreted as relevant input.
> Input files must have an extension ".ini".

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*         UATU notes           *
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

> This has been copied from the lcdm.ini file in class. It will replace the values of Omega_c, A_s, and input/output
> locations for use with Uatu mPk's. All others will be left as defaults. 

----------------------------
----> background parameters:
----------------------------

1) Hubble parameter : either 'H0' in km/s/Mpc or 'h' (default: 'h' set to 0.704)

#H0 = 72.
h =0.7

2) photon density: either 'T_cmb' in K or 'Omega_g' or 'omega_g' (default: 'T_cmb' set to 2.726)

T_cmb = 2.726
#Omega_g = 5.050601e-05
#omega_g = 2.47479449e-5

3) baryon density: either 'Omega_b' or 'omega_b' (default: 'omega_b' set to 0.02253)

Omega_b = 0.022
#omega_b = 0.0266691

4) ultra-relativistic species / massless neutrino density: either 'N_eff' or 'Omega_ur' or 'omega_ur' (default: 'N_eff' set to 3.04)

N_eff = 3.04
#Omega_ur = 3.4869665e-05
#omega_ur = 1.70861334e-5

5) density of cdm (cold dark matter): 'Omega_cdm' or 'omega_cdm' (default: 'omega_cdm' set to 0.1122)

Omega_cdm = {ocdm:f}
#omega_cdm = 0.110616

6) cosmological constant / fluid with constant w and sound speed (can be used to model simple dark energy models): enter one parameter out of 'Omega_Lambda' or 'Omega_fld', the other one is inferred by the code in such way that (sum_i Omega_i) equals (1 + Omega_k) (default: 'Omega_fld' set to 0 and 'Omega_Lambda' to (1+Omega_k-Omega_tot))

# Omega_Lambda = 0.7
Omega_fld = 0

7) equation of state parameter (p/rho equal to w0+wa(1-a0/a) ) and sound speed of the fluid (this is the sound speed defined in the frame comoving with the fluid, i.e. obeying to the most sensible physical definition)

w0_fld = -1.0
#w0_fld = -0.9
wa_fld = 0.
cs2_fld = 1

--------------------------------
----> thermodynamics parameters:
--------------------------------

1) primordial Helium fraction 'YHe' (default: set to 0.25)

YHe = 0.25

2) enter one of 'z_reio' or 'tau_reio' (default: 'z_reio' set to 10.3)

z_reio = 10.
#tau_reio = 0.084522

----------------------------------------------------
----> define which perturbations should be computed:
----------------------------------------------------

1) list of output spectra requested ('tCl' for temperature Cls, 'pCl' for polarization CLs, 'lCl' for lensing potential Cls, , 'dCl' for matter density Cls, 'mPk' for total matter power spectrum P(k) infered from gravitational potential, 'mTk' for matter transfer functions for of each species). Can be attached or separated by arbitrary characters. Given this list, all non-zero auto-correlation and cross-correlation spectra will be automatically computed. Can be left blank if you do not want to evolve cosmological perturbations at all. (default: set to blanck, no perturbation calculation)

#output = tCl,pCl,lCl
#output = tCl,pCl,lCl,mPk
output = mPk

2) relevant only if you ask for scalars, temperature or polarisation Cls, and 'lCl': if you want the spectrum of lensed Cls, enter a word containing the letter 'y' or 'Y' (default: no lensed Cls)

lensing = no

---------------------------------------------
----> define primordial perturbation spectra:
---------------------------------------------

1) pivot scale in Mpc-1 (default: set to 0.002)

k_pivot = 0.05

2) scalar adiabatic perturbations: curvature power spectrum value at pivot scale, tilt at the same scale, and tilt running (default: set 'A_s' to 2.42e-9, 'n_s' to 0.967, 'alpha_s' to 0)

A_s = {as:f}
n_s = 0.96
alpha_s = 0.

-------------------------------------
----> define format of final spectra:
-------------------------------------

1) maximum l 'l_max_scalars', 'l_max_tensors' in Cls for scalars/tensors (default: set 'l_max_scalars' to 2500, 'l_max_tensors' to 500)

l_max_scalars = 3000
l_max_tensors = 500

2) maximum k in P(k), 'P_k_max_h/Mpc' in units of h/Mpc or 'P_k_max_1/Mpc' in units of 1/Mpc. If scalar Cls are also requested, a minimum value is automatically imposed (the same as in scalar Cls computation) (default: set to 0.1h/Mpc)

P_k_max_h/Mpc = 10.
#P_k_max_1/Mpc = 0.7

3) value(s) 'z_pk' of redshift(s) for P(k,z) output file(s); can be ordered arbitrarily, but must be separated by comas (default: set 'z_pk' to 0)

z_pk = 20.
#z_pk = 0., 1.2, 3.5

4) if you need Cls for the matter density autocorrelation or cross density-temperature correlation (option 'dCl'), enter here a description of the selection functions W(z) of each redshift bin; selection can be set to 'gaussian', then pass a list of N mean redshifts in growing order separated by comas, and finally 1 or N widths separated by comas (default: set to 'gaussian',1,0.1)

selection=gaussian
selection_mean = 1.
selection_width = 0.5

5) file name root 'root' for all output files (default: set 'root' to 'output/') (if Cl requested, written to '<root>cl.dat'; if P(k) requested, written to '<root>pk.dat'; plus similar files for scalars, tensors, pairs of initial conditions, etc.; if file with input parameters requested, written to '<root>parameters.ini')

root = {output_root} 

6) do you want headers at the beginning of each output file (giving precisions on the output units/ format) ? If 'headers' set to something containing the letter 'y' or 'Y', headers written, otherwise not written (default: written)

headers = no

7) in all output files, do you want columns to be normalized and ordered with the default CLASS definitions or with the CAMB definitions (often idential to the CMBFAST one) ? Set 'format' to either 'class', 'CLASS', 'camb' or 'CAMB' (default: 'class')

format = camb

8) if 'bessel file' set to something containing the letters 'y' or 'Y', the code tries to read bessel functions in a file; if the file is absent or not adequate, bessel functions are computed and written in a file. The file name is set by defaut to 'bessels.dat' but can be changed together with precision parameters: just set 'bessel_file_name' to '<name>' either here or in the precision parameter file. (defaut: 'bessel file' set to 'no' and bessel functions are always recomputed)

bessel file = no

9) Do you want to have all input/precision parameters which have been read written in file '<root>parameters.ini', and those not written in file '<root>unused_parameters' ? If 'write parameters' set to something containing the letter 'y' or 'Y', file written, otherwise not written (default: not written)

write parameters = yeap

----------------------------------------------------
----> amount of information sent to standard output:
----------------------------------------------------

Increase integer values to make each module more talkative (default: all set to 0)

background_verbose = 1
thermodynamics_verbose = 0
perturbations_verbose = 1
bessels_verbose = 0
transfer_verbose = 0
primordial_verbose = 1
spectra_verbose = 0
nonlinear_verbose = 1
lensing_verbose = 0
output_verbose = 1

"""
# TODO have to fix the baryon numbers, they're wrong!!!
picola_config = """
% =============================== %
% This is the run parameters file %
% =============================== %

% Simulation outputs
% ==================
OutputDir                   {outputdir}                               % Directory for output.
FileBase                    {file_base} % Base-filename of output files (appropriate additions are appended on at runtime)
OutputRedshiftFile          {outputdir}/output_redshifts.dat           % The file containing the redshifts that we want snapshots for
NumFilesWrittenInParallel   4                                    % limits the number of files that are written in parallel when outputting.

% Simulation Specifications
% =========================
UseCOLA          1           % Whether or not to use the COLA method (1=true, 0=false).
Buffer           1.3         % The amount of extra memory to reserve for particles moving between tasks during runtime.
Nmesh            512         % This is the size of the FFT grid used to compute the displacement field and gravitational forces.
Nsample          512         % This sets the total number of particles in the simulation, such that Ntot = Nsample^3.
Box              512.0      % The Periodic box size of simulation.
Init_Redshift    20.0         % The redshift to begin timestepping from (redshift = 9 works well for COLA)
Seed             5001        % Seed for IC-generator
SphereMode       0           % If "1" only modes with |k| < k_Nyquist are used to generate initial conditions (i.e. a sphere in k-space),
                             % otherwise modes with |k_x|,|k_y|,|k_z| < k_Nyquist are used (i.e. a cube in k-space).

WhichSpectrum    1           % "0" - Use transfer function, not power spectrum
                             % "1" - Use a tabulated power spectrum in the file 'FileWithInputSpectrum'
                             % otherwise, Eisenstein and Hu (1998) parametrization is used
                             % Non-Gaussian case requires "0" and that we use the transfer function

WhichTransfer    0           % "0" - Use power spectrum, not transfer function
                             % "1" - Use a tabulated transfer function in the file 'FileWithInputTransfer'
                             % otherwise, Eisenstein and Hu (1998) parameterization used
                             % For Non-Gaussian models this is required (rather than the power spectrum)

FileWithInputSpectrum  {outputdir}/class_pk.dat    % filename of tabulated input spectrum (if used)
                                                   % expecting k and Pk

FileWithInputTransfer  /home/users/swmclau2/picola/l-picola/files/input_transfer.dat    % filename of tabulated transfer function (if used)
                                                   % expecting k and T (unnormalized)

% Cosmological Parameters
% =======================
Omega            {ocdm:.2f}        % Total matter density (CDM + Baryons at z=0).
OmegaBaryon      {ob:.2f}        % Baryon density (at z=0).
OmegaLambda      {olambda:.2f}        % Dark Energy density (at z=0)
HubbleParam      0.7         % Hubble parameter, 'little h' (only used for power spectrum parameterization).
Sigma8           {sigma8:.2f}        % Power spectrum normalization (power spectrum may already be normalized correctly).
PrimordialIndex  0.96        % Used to tilt the power spectrum for non-tabulated power spectra (if != 1.0 and nongaussian, generic flag required)

% Timestepping Options
% ====================
StepDist         0           % The timestep spacing (0 for linear in a, 1 for logarithmic in a)
DeltaA           0           % The type of timestepping: "0" - Use modified COLA timestepping for Kick and Drift. Please choose a value for nLPT.
                             % The type of timestepping: "1" - Use modified COLA timestepping for Kick and standard Quinn timestepping for Drift. Please choose a value for nLPT.
                             % The type of timestepping: "2" - Use standard Quinn timestepping for Kick and Drift
                             % The type of timestepping: "3" - Use non-integral timestepping for Kick and Drift
nLPT             -2.5        % The value of nLPT to use for modified COLA timestepping


% Units
% =====
UnitLength_in_cm                3.085678e24       % defines length unit of output (in cm/h)
UnitMass_in_g                   1.989e43          % defines mass unit of output (in g/h)
UnitVelocity_in_cm_per_s        1e5               % defines velocity unit of output (in cm/sec)
InputSpectrum_UnitLength_in_cm  3.085678e24       % defines length unit of tabulated input spectrum in cm/h.
                                                  % Note: This can be chosen different from UnitLength_in_cm

% ================================================== %
% Optional extras (must comment out if not required) %
% ================================================== %

% Non-Gaussianity
% ===============
%Fnl                  -400                              % The value of Fnl.
%Fnl_Redshift         49.0                              % The redshift to apply the nongaussian potential
%FileWithInputKernel  /files/input_kernel_ortog.txt     % the input kernel for generic non-gaussianity (only needed for GENERIC_FNL)


% Lightcone simulations
% =====================
%Origin_x     0.0                % The position of the lightcone origin in the x-axis
%Origin_y     0.0                % The position of the lightcone origin in the y-axis
%Origin_z     0.0                % The position of the lightcone origin in the z-axis
%Nrep_neg_x   0                  % The maximum number of box replicates in the negative x-direction
%Nrep_pos_x   0                  % The maximum number of box replicates in the positive x-direction
%Nrep_neg_y   0                  % The maximum number of box replicates in the negative y-direction
%Nrep_pos_y   0                  % The maximum number of box replicates in the positive y-direction
%Nrep_neg_z   0                  % The maximum number of box replicates in the negative z-direction
%Nrep_pos_z   0                  % The maximum number of box replicates in the positive z-direction
"""
