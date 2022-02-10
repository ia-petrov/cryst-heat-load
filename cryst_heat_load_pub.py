#heat propagation in crystals under XFEL pulses and their diffraction in heated crystals
#I. Petrov, European XFEL, August 2021, email: ilia.petrov@xfel.eu
#heat flow calculations based on H. Sinn's code 27.7.2007 DESY (github.com/haraldsinn/pulses_slab)
#diffraction based on Bushuev//BRAS Phys 77 (2013)
#this code is a part of I. Petrov et al.// arXiv:2112.01826v1

import numpy as np
from matplotlib import pyplot as plt
from math import *
from numba import njit,jit,prange
from scipy import special,signal,integrate, ndimage
from scipy import optimize as opt






#get_ipython().run_line_magic('matplotlib', 'notebook')

en_ev0 = 9000.0

def R_for_deform(d_en,
                 deform_T,
                 en_ev0,
                 beam_int,
                 pol):
    """
    calculates diffraction in heated crystal
    d_en     - photon energy difference from the central photon energy
    deform_T - relative lattice deformation arrray along a selected coordinate
    en_ev0   - central photon energy
    beam_int - intensity distribution along a selected coordinate
    pol      - polarization 'sigma' or 'pi'
    """

    #en_ev0 9keV
    chi0_0 = -0.12073e-4+1j*0.22532e-6
    chi_h_0 = -0.63776e-5+1j*0.15706e-6
    d_sp = 3.135e-4
    thick_dif = 100
    C_pol=1

    en_ev = en_ev0+d_en
    wavel0 = 12398.0/en_ev0*1e-4

    chi0 = chi0_0*(1+(en_ev-en_ev0)**2/en_ev0**2)
    chi_h = chi_h_0*(1+(en_ev-en_ev0)**2/en_ev0**2)
    chimh = chi_h

    wavel = 12398/en_ev*1e-4
    th_B = np.arcsin(wavel0/2/d_sp)
    if (pol=='pi'):
        C_pol= np.cos(2*th_B)
    
    gamma0 = np.sin(th_B)
    gammag = np.sin(th_B)
    theta = th_B+1.15/9000*np.tan(th_B)
    K_wave = 2*np.pi/wavel
    alpha = 2*np.sin(2*th_B)*(theta-th_B+np.tan(th_B)*((en_ev-en_ev0)/en_ev0+deform_T))
    b=-gamma0/gammag
    root = np.sqrt((chi0*(1.0-b)-alpha*b)**2+4.0*b*C_pol*C_pol*chi_h*chimh)
    eps1 = 1/gamma0/4*(chi0*(1.0+b)+alpha*b+root)
    g1 = np.exp(1j*K_wave*eps1*thick_dif)
    eps1=None
    eps2 = 1/gamma0/4*(chi0*(1.0+b)+alpha*b-root)
    g2 = np.exp(1j*K_wave*eps2*thick_dif)
    eps2=None
    alpha1 = alpha*b-chi0*(1-b)
    alpha=None
    R1 = (alpha1+root)/2/C_pol/chimh
    R2 = (alpha1-root)/2/C_pol/chimh
    R = (R1-R1*g1/g2)/(1-R1*g1/R2/g2)
    R1=None
    R2=None
    g1=None
    g2=None

    return beam_int*np.abs(R)**2
#     return R



def depth_flow_heat(T_pulse_z_in,
                    lambda_arr,
                    cp_arr,
                    iter_num,
                    dt,
                    T_arr,
                    T_init,
                    dz,
                    x_area,
                    rho,
                    x_vol,
                    transm_att,
                    XGM_trend,
                    xz_heat_rel_x,
                    x_i,
                    ip_arr, heat_int, heat_0): # These were global in the original
    """
    calculates the temperature distribution along depth after the propagation of heat for all pulses in a train in a single cylinder section
    T_pulse_z     - initial temperature distribution
    lambda_arr    - array of thermal conductivity for the array of temperatures in args[5]
    cp_arr        - array of specific heat for the array of temperatures in args[5]
    iter_num      - number of time steps between two pulses
    dt            - duration of each time step
    T_arr         - array of temperatures for thermal conductivity and specific heat
    T_init        - initial temperature
    dz            - step in depth direction
    x_area        - array of areas of rings along radius
    rho           - material density
    x_vol         - array of volumes of cylinder sections along radius
    transm_att     - attenuation factor
    XGM_trend      - energies of all pulse
    xz_heat_rel_x  - fraction of heat absorbed in a disc
    x_i            - the index of radial coordinate for a given cylinder section
    """

    T_pulse_surf_x = np.zeros(len(ip_arr))
    
    # Do not overwrite input array
    T_pulse_z = T_pulse_z_in.copy()
    
    for ip in ip_arr[1:]:
        jiter=0
        while (jiter<iter_num):
            lambda_pulse = np.interp(T_pulse_z,T_arr,lambda_arr)
            cp_pulse = np.interp(T_pulse_z,T_arr,cp_arr)
            T_pulse_z[-1]=T_init
            #like in Harald's code
            #j_left and j_right are streams at left and right edges in the direction to the left
            T_pulse_m1 = np.roll(T_pulse_z,1)
            j_up=(T_pulse_m1-T_pulse_z)*lambda_pulse/dz
            j_up=j_up#*x_area
            j_down=np.roll(j_up,-1)
            j_up[0]=0
            j_down[-1]=0
            dT_dt =(j_up-j_down)/(cp_pulse*rho*dz)#*x_area)
#             if (x_i==0)&(ip==1)&(jiter in [1,2,3,4,5,6,7,8,9,10,11,12]):
#                 print(jiter,dT_dt[0]*dt*1e2,T_pulse_z[0])
            T_pulse_z+=dT_dt*dt*1e2
            
            jiter+=1
#         deform_T = np.interp(T_pulse_z[0],T_arr,deform_int)-deform_init
        pulse_energy = XGM_trend[ip]*transm_att*1e-6#+120e-6
        heat_pulse_ip_z = np.interp(T_pulse_z,T_arr,heat_int)-heat_0
        T_pulse_surf_x[ip] = T_pulse_z[0]
        heat_pulse_ip_z+=xz_heat_rel_x*pulse_energy
        T_pulse_z = np.interp(heat_pulse_ip_z,heat_int-heat_0,T_arr)
    
    return T_pulse_z, T_pulse_surf_x 
