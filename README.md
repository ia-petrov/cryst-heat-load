The python script calculates the transmission of double-bounce monochromator illuminated by a train of XFEL pulses. The methodology is described in [1]. The published notebook calculates the transmission for the parameters in [1].

The .dat files contain the following parameters of silicon:
"alpha_si.dat": linear expansion coefficient [2]
"lambda_si_28.dat": thermal conductivity [3] 
"cp_silicon_debye.dat": the specific heat calculated by the Debye's model

The user needs to define:

choice tuple for material density: rho_choice; for silicon it would be {"si": 2.33}\
material selection: material = "si"

energy of each pulse in a train in J: pulse_energy\
FWHM size of the pulse in microns: FWHM_norm\
central photon energy of the pulses in eV: en_ev0\
number of pulses in a train: pul_in_tr = 50\
absortion length of the crystal at the selected conditions in microns: abs_len\
initial temperature in K: T_init

repetition rate in MHz: reprate (the time separation between pulses in microseconds would be 1/reprate)

in R_for_deform function one needs to define the diffraction parameters:\ 
suscpetibilities: chi0_0 and chi_h_0\
lattice spacing in A: d_sp\
crystal thickness for diffraction "thick_dif", which needs to be several extinction lengths\
polarization factor (1 for sigma): C_pol=1

optionally one can define the energy of each pulse in a train in the array XGM_trend, by default all pulses have the same energy

The script creates three plots:

1) the temperature distribution at surface along radius after various number of pulses "T_profile.png"\
2) rocking curves after various number of pulses "RCs.png"\
3) the transmission of the monochromator during train "transm_train.png"

the user needs to adjust the variable "thick", which defines the location of the heat sink along the depth coordinate. "thick" has to be such that after its increase the code provides the same results. It is recommended to first define the parameters of the pulse and crystals, and then, starting from a small thickness, gradually increase the thickness until the output plots stop changing.

iter_num defines the number of time steps when calculating heat flow. It needs to be large enough such that no numerical errors occur. iter_num=5000 was sufficient for the parametrs in the published notebook

multiprocessing is implemented to minimize calculation time, this might require installing additional python libraries

[1] I. Petrov et al.// "Performance of a cryo-cooled crystal monochromator illuminated by hard X-rays with MHz repetition rate at the European X-ray Free-Electron Laser, arXiv:2112.01826v1.\
[2] Y. Okada et al//J. Appl. Phys.56, 314–320 (1984)\
[3] Glassbrenner et al// Phys. Rev.134, A1058–A1069 (1964)

