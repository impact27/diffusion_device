# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:47:11 2017

@author: quentinpeter
"""
import diffusion_device.profile as dp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
import os
from image_registration.image import is_overexposed
from diffusion_device.json import full_fit

#==============================================================================
# Settings
#==============================================================================

#File Name.
settingsfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_fitSettings.json'
outpath = 'output'
plotpos = [0, 10, 15]
#==============================================================================
# Fit----------DO NOT CHANGE BELOW THIS POINT----------------------------------
#==============================================================================
#        _  _
#  ___ (~ )( ~)
# /   \_\ \/ /
#|   D_ ]\ \/
#|   D _]/\ \
# \___/ / /\ \
#      (_ )( _)

# Infer variables    
(radii, profiles_list, 
 fits_list, LSE, pixs, overexposed) = full_fit(settingsfn)

overexposed=[is_overexposed(im) for im in ims]
#%%


intensity = np.asarray([ np.nanmax(p) for p in profiles_list])
radii = np.asarray(radii)
LSE = np.asarray(LSE)
pixs = np.asarray(pixs)

if outpath is not None:
    base_name = os.path.join(outpath, 
                             os.path.splitext(os.path.basename(settingsfn))[0])

#%%
x=np.arange(len(radii))
valid=np.logical_not(overexposed)
figure()
plot(x[valid],radii[valid]*1e9,'x',label='data')
plt.xlabel('Frame number')
plt.ylabel('Radius [nm]')
if np.any(overexposed):
    plot(x[overexposed],radii[overexposed]*1e9,'x',label='overexposed data')
    plt.legend()
if outpath is not None:
    plt.savefig(base_name+'_R_fig.pdf')
    
figure()
plot(x[valid],LSE[valid],'x',label='regular')
plt.xlabel('Frame number')
plt.ylabel('Least square error')
if np.any(overexposed):
    plot(x[overexposed],LSE[overexposed],'x',label='overexposed')
    plt.legend()
if outpath is not None:
    plt.savefig(base_name+'_LSE_fig.pdf')
    
figure()
plot(x[valid],intensity[valid],'x',label='regular')
plt.xlabel('Frame number')
plt.ylabel('Maximum intensity')
if np.any(overexposed):
    plot(x[overexposed],intensity[overexposed],'x',label='overexposed')
    plt.legend()
if outpath is not None:
    plt.savefig(base_name+'_max_intensity_fig.pdf')
    
figure()
plot(x,pixs*1e6,'x')
plt.xlabel('Frame number')
plt.ylabel('Pixel size')
if outpath is not None:
    plt.savefig(base_name+'_pixel_size_fig.pdf')
    


if outpath is not None:
    import shutil
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outsettingsfn = os.path.join(outpath, os.path.basename(settingsfn))
    shutil.copy(settingsfn, outsettingsfn)  
    with open(base_name+'_result.txt','wb') as f:
        f.write('Radii:\n'.encode())
        np.savetxt(f,radii)
        f.write('Least square error:\n'.encode())
        np.savetxt(f,LSE)
        if np.any(overexposed):
            f.write('Overexposed Frames:\n'.encode())
            np.savetxt(f,overexposed)
        f.write('Pixel size:\n'.encode())
        np.savetxt(f,pixs)
        

        
#%%
plotpos=np.asarray(plotpos)

for pos in plotpos[plotpos<len(profiles_list)]:
    
    profs=dp.get_fax(profiles_list[pos])
    X=np.arange(len(profs))*pixs[pos]*1e6
    figure()
    plot(X,profs)
    
    fits=dp.get_fax(fits_list[pos])

    plot(X,fits)
    plt.title('r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
            radii[pos]*1e9, LSE[pos], pixs[pos]*1e6))   
    plt.xlabel('Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')
