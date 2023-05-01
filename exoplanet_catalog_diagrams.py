#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:55:28 2023

@author: mam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_earth, M_jup, M_sun, R_earth, R_jup, R_sun, h, c, k_B
from astropy import units as u

def black_body(temp, wl, h, c, k_B):
    """
    Power emitted per unit area per unit solid angle per level of energy (channel spectrum)
    in W/m2/sr/m.
    This is also the power through a unit of optical étendue per level of energy.
    """
    return 2 * h * c / wl.to(u.m) * 1/wl.to(u.m)**2 * c/wl.to(u.m)/wl.to(u.m) * 1 / (np.exp(h * c / (wl.to(u.m) * k_B * temp)) - 1)


def get_planet_radius(data_list):
    """ Extract or calculate the radii of the exoplanets in the list.
    Output in Jupiter radius"""    

    # Load radius, some values may be NaN
    radius = data_list['radius']
    # Identify Nan
    nan_idx = data_list[data_list['radius'].apply(np.isnan)].index
    
    # For missing radius, let's deduce from the mass
    mass = data_list.loc[nan_idx, 'mass']
    data_list.loc[nan_idx, 'radius'] = mass**(1/3) # assuming Jupiter-type planet
    nan_idx = data_list[data_list['radius'].apply(np.isnan)].index
    radius = data_list['radius']
    
    return radius, nan_idx


def get_planet_teff(data_list, s_teff, albedo=0.3):
    """ Extract or calculate the effective temperature of the exoplanets in the list.
    Output in Kelvin"""    

    # Load temperature, some values may be NaN
    nan_idx = data_list[data_list['temp_measured'].apply(np.isnan)].index
    # Use the calculated temperature where NaN are
    data_list.loc[nan_idx, 'temp_measured'] = data_list['temp_calculated']
    nan_idx = data_list[data_list['temp_measured'].apply(np.isnan)].index
    
    # Now we calculate the temperature if it is not in the list
    # We assume the flux received by the planet is equal to its radiative energy
    au = u.au.to(u.m)
    p_sma = data_list.loc[nan_idx, 'semi_major_axis']
    s_rad = data_list.loc[nan_idx, 'star_radius']
    # p_sma_nan_idx = data_list[data_list['semi_major_axis'].apply(np.isnan)].index
    # s_rad_nan_idx = data_list[data_list['star_radius'].apply(np.isnan)].index
    
    p_teff = s_teff * ((s_rad * R_sun) / (2 * p_sma * au))**0.5 * (1 - albedo)**0.25
    data_list.loc[nan_idx, 'temp_measured'] = p_teff
    nan_idx = data_list[data_list['temp_measured'].apply(np.isnan)].index

    # Return the effective temperatures (measured, calculated, estimated)
    data_list.loc[data_list['temp_measured'] < 100, 'temp_measured'] = 100
    data_list.loc[data_list['temp_measured'] > 2500, 'temp_measured'] = 2500
    t_eff = data_list['temp_measured']

    return t_eff, nan_idx, data_list['semi_major_axis'], data_list['star_distance']

def get_star_radius(data_list):
    """ Extract or calculate the radii of the stars in the list.
    Output in Sun radius"""  
    
    # Load radius, some values may be NaN
    radius = data_list['star_radius']
    # Identify Nan
    nan_idx_radius = data_list[data_list['star_radius'].apply(np.isnan)].index
    
    return radius, nan_idx_radius

def get_star_teff(data_list, sp_file):
    """ Extract or calculate the effective temperature of the stars in the list.
    Output in Kelvin.
    Estimations are based on Allen"""

    # Load effective temperatures, some values may be NaN
    t_eff = data_list['star_teff']
    
    # Identify Nan
    nan_idx = data_list[data_list['star_teff'].apply(np.isnan)].index
    
    # Use the spectral type
    spectral_type = data_list.loc[nan_idx, 'star_sp_type']
    nan_idx_sp = []
    
    for k in spectral_type.index:
        try:
            sp = spectral_type.loc[k]
            try: # to avoid NaN
                sp = sp.replace(' ', '')
                sp = sp.lower()
                if '/' in sp or '-' in sp:
                    sp = reformat_string(sp)
                mask = sp_file['SpT'].str.contains(sp)
                sp_idx = np.where(mask==True)[0]
                # Check if the spectral type is in the catalog
                if sp_idx.size == 0: # it is absent
                    sp = sp[:2] # Probably a complex spectral type, we just keep the bare minimum
                    mask = sp_file['SpT'].str.contains(sp)
                    sp_idx = np.where(mask==True)[0]
                    # Check again
                    if sp_idx.size == 0:
                        nan_idx_sp.append(k)
                else:
                    temperature = sp_file['Teff'][sp_idx[0]].values[0]
                    data_list.loc[k, 'star_teff'] = temperature
            except AttributeError:
                nan_idx_sp.append(k)
        except TypeError:
            nan_idx_sp.append(k)

    t_eff = data_list['star_teff']
    return t_eff, nan_idx_sp

def reformat_string(string):
    if '/' in string:
        symbol = '/'
    else:
        symbol = '-'
        
    idx = string.index(symbol)
    if 'v' in string[idx:idx+2]:
        string = string.replace(string[idx-2:idx+1], '')
        return string
    if 'iii' in string[:idx]:
        return string[:idx]

def calculate_contrast(data_list, sp_file, wav):
    s_teff, bad_idx_teff = get_star_teff(data_list, sp_file)
    s_rad, bad_idx_s_rad = get_star_radius(data_list)
    p_rad, bad_idx_p_rad = get_planet_radius(data_list)
    p_teff, bad_idx_p_teff, p_sma, s_dist = get_planet_teff(data_list, s_teff) 
    
    s_flx = black_body(s_teff, wav, h, c, k_B)
    p_flx = black_body(p_teff, wav, h, c, k_B)
    contrast = p_flx / s_flx * (p_rad * R_jup)**2 / (s_rad * R_sun)**2
    
    # Calculate the angular distance from the host star to the planet
    ang_dist = p_sma / s_dist
    
    return contrast, ang_dist, p_sma

wav = 3.8e-6 * u.m
path = ''
data_rv = pd.read_csv(path+'exoplanet.eu_catalog_rv.csv')
data_microlensing = pd.read_csv(path+'exoplanet.eu_catalog_microlensing.csv')
data_imaging = pd.read_csv(path+'exoplanet.eu_catalog_imaging.csv')
data_astrometry = pd.read_csv(path+'exoplanet.eu_catalog_astrometry.csv')
data_transit = pd.read_csv(path+'exoplanet.eu_catalog_transit.csv')
sp_file = pd.read_csv(path+'list_spectral_types.csv')
sp_file['SpT'] = sp_file['SpT'].str.lower()

# ratio_M = M_jup / M_earth
# width = 10
# height = width / 1.618
# fig = plt.figure(figsize=(width, height))
# ax = plt.subplot(111)
# ax.scatter(data_transit['semi_major_axis'], data_transit['mass']*ratio_M, label='transit', alpha=0.25)
# ax.scatter(data_rv['semi_major_axis'], data_rv['mass']*ratio_M, label='RV')
# ax.scatter(data_microlensing['semi_major_axis'], data_microlensing['mass']*ratio_M, label='microlensing', alpha=0.5)
# ax.scatter(data_astrometry['semi_major_axis'], data_astrometry['mass']*ratio_M, label='astrometry', alpha=0.5)
# ax.scatter(data_imaging['semi_major_axis'], data_imaging['mass']*ratio_M, label='imaging', marker='^', s=50)
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_ylim(5e-5, 150)
# ax.legend(loc='best', fontsize=20)
# ax.set_xlabel('Distance to star (AU)', size=20)
# ax.set_ylabel('Mass (Mearth)', size=20)
# ax.tick_params(axis='both', which='major', labelsize=18)
# plt.tight_layout()
# plt.savefig('exoplanets_mass_vs_semimajoraxis.png', format='png', dpi=150)

# Values of coronagraphy contrast curve vs angular resolution from Absil et al. 2016
x = 1000 * np.array([0.09298999, 0.12732475, 0.23748212, 0.3090129, 0.40772533, 0.50214595, 0.6180258, 0.73819745, 0.88412017, 0.99427754, 0.99427754])
y = 10**(np.array([4.9503484, 5.7644954, 7.350796, 8.110622, 9.080728, 9.696287, 10.257017, 10.799459, 11.264546, 11.504548, 11.504548])/(-2.5))
x = 1000 * np.array([0.20, 0.3090129, 0.40772533, 0.50214595, 0.6180258, 0.73819745, 0.88412017, 0.99427754, 0.99427754])
y = 10**(np.array([7.050796, 8.110622, 9.080728, 9.696287, 10.257017, 10.799459, 11.264546, 11.504548, 11.504548])/(-2.5)) / 5

# Values of NRM contrast curves vs angular resolution from Hinckley 2009
x2 = np.array([30.656874, 25.482527 ,40.733788  ,  47.645367, 60.59634  ,92.46987,  120.74288 ,151.2749 ,179.06439  ,203.16444  ,240.63087  ,269.2494])
y2 = 1./np.array([287.66772, 178.99178,  568.15344,  913.3422, 1243.229  ,1431.6403, 1261.3909,  1367.5122,  1310.062, 1204.5841 ,1333.9159  ,1278.5238])/5.

# Values from Hinckley 2009
x3 = np.array([0.7,10])
y3 = np.array([2.e-3,2.e-3])

rv_contrast, rv_dist, rv_sma = calculate_contrast(data_rv, sp_file, wav)
microlensing_contrast, microlensing_dist, microlensing_sma = calculate_contrast(data_microlensing, sp_file, wav)
imaging_contrast, imaging_dist, imaging_sma = calculate_contrast(data_imaging, sp_file, wav)
astrometry_contrast, astrometry_dist, astrometry_sma = calculate_contrast(data_astrometry, sp_file, wav)
transit_contrast, transit_dist, transit_sma = calculate_contrast(data_transit, sp_file, wav)

mas = 1000
rv_dist *= mas
microlensing_dist *= mas
imaging_dist *= mas
astrometry_dist *= mas
transit_dist *= mas

vlt_res = np.degrees(wav.value/8.) * 3600 * mas
elt_res = np.degrees(wav.value/39.) * 3600 * mas
vlti_res = np.degrees(wav.value/140.) * 3600 * mas
chara_res = np.degrees(wav.value/330.) * 3600 * mas

text_sz = 14
width = 10
height = width / 1.618
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
ylim = (1e-8, 1)
xlim = (1e-3*mas, 1*mas)

rv_mask = np.where((rv_dist >= xlim[0])&(rv_dist <= xlim[1]))
microlensing_mask = np.where((microlensing_dist >= xlim[0])&(microlensing_dist <= xlim[1]))
imaging_mask = np.where((imaging_dist >= xlim[0])&(imaging_dist <= xlim[1]))
astrometry_mask = np.where((astrometry_dist >= xlim[0])&(astrometry_dist <= xlim[1]))
transit_mask = np.where((transit_dist >= xlim[0])&(transit_dist <= xlim[1]))

# fig = plt.figure(figsize=(width, height))
# ax = plt.subplot(111)
# ax.scatter(transit_dist, transit_contrast, label='transit', marker='*', alpha=0.25)
# ax.scatter(rv_dist, rv_contrast, label='RV')
# ax.scatter(microlensing_dist, microlensing_contrast, label='microlensing', marker='s', alpha=0.5)
# ax.scatter(astrometry_dist, astrometry_contrast, label='astrometry', marker='x', alpha=0.5)
# ax.scatter(imaging_dist, imaging_contrast, label='imaging', marker='^', s=50)
# ax.plot(x, y, c=colours[0])
# ax.plot(x2, y2, ':', c=colours[1])
# ax.axvline(vlt_res, ls='--', c=colours[5])
# ax.axvline(elt_res, ls='--', c=colours[5])
# ax.axvline(vlti_res/2, ls='--', c=colours[5])
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(*ylim)
# ax.set_xlim(*xlim)
# ax.text(0.25*mas, 1.5e-6, 'Coronography\n (10 m)', rotation=-30, multialignment='center', fontsize=text_sz, c=colours[0])
# ax.text(0.03*mas, 0.45e-3, 'NRM', rotation=-30, fontsize=text_sz, c=colours[1])
# ax.text(vlt_res*0.56, 1.05, r"$\lambda$/D"+"\n"+"(3.8$\mu$m, D=8m)", multialignment='center', fontsize=text_sz, c=colours[5])
# ax.text(elt_res*0.53, 1.05, '$\lambda$/D\n (3.8$\mu$m, D=39m)', multialignment='center', fontsize=text_sz, c=colours[5])
# ax.text(vlti_res/2*0.53, 1.05, '$\lambda$/2B\n (3.8$\mu$m, D=8m)', multialignment='center', fontsize=text_sz, c=colours[5])
# ax.fill_between((vlti_res/2, xlim[1]), 1e-3, 1, alpha=0.3, facecolor=colours[3])
# ax.fill_between((vlt_res, xlim[1]), 5e-6, 1e-3, alpha=0.3, facecolor=colours[3])
# ax.legend(loc='upper left', fontsize=text_sz, ncol=2)
# ax.set_xlabel('Distance to star (mas)', size=20)
# ax.set_ylabel('Contrast', size=20)
# ax.tick_params(axis='both', which='major', labelsize=16)
# plt.tight_layout()

# text_sz = 14
# width = 10
# height = width / 1.618
# colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
# ylim = (1e-8, 1)
# xlim = (1e-3*mas, 1*mas)
# fig = plt.figure(figsize=(width, height))
# ax = plt.subplot(111)
# ax.scatter(transit_sma.loc[transit_mask], transit_contrast.loc[transit_mask], label='transit', marker='*', alpha=0.6)
# ax.scatter(rv_sma.loc[rv_mask], rv_contrast.loc[rv_mask], label='RV')
# ax.scatter(imaging_sma.loc[imaging_mask], imaging_contrast.loc[imaging_mask], label='imaging', marker='^', s=75)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(*ylim)
# ax.legend(loc='upper left', fontsize=text_sz, ncol=2)
# ax.set_xlabel('Distance to star (AU)', size=20)
# ax.set_ylabel('Contrast', size=20)
# ax.tick_params(axis='both', which='major', labelsize=16)
# plt.tight_layout()

text_sz = 16
width = 10
height = width / 1.618
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
ylim = (1e-8, 1)
xlim = (1e-3*mas, 1*mas)
fig = plt.figure(figsize=(width, height))
ax = plt.subplot(111)
ax.scatter(transit_sma, transit_contrast, label='Transit', marker='*', alpha=0.5)
ax.scatter(rv_sma, rv_contrast, label='RV')
ax.scatter(imaging_sma, imaging_contrast, label='Imaging', marker='^', s=75)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(*ylim)
ax.legend(loc='upper left', fontsize=text_sz, ncol=2)
ax.set_xlabel('Distance to star (AU)', size=20)
ax.set_ylabel('Contrast @ %.1f $\mu$m'%(wav.to(u.um).value), size=20)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
# plt.savefig('exoplanets_contrast_vs_semimajoraxis.png', format='png', dpi=150)
