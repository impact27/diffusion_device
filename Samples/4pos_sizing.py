"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
import diffusion_device.profile as dp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
from diffusion_device.json import full_fit
#==============================================================================
# Settings
#==============================================================================

#Settings file location
settingsfn='../diffusion_device/tests/test_data/UVim300ulph_fitSettings_poly_all.json'
settingsfn='../diffusion_device/tests/test_data/Brightim900ulph_fitSettings.json'

#Where to save the run
outpath = 'output'


#==============================================================================
# Fit
#==============================================================================

radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)

if outpath is not None:
    import os
    import shutil
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    base_name = os.path.join(outpath,  
                             os.path.splitext(os.path.basename(settingsfn))[0])
    shutil.copy(settingsfn, base_name + '.json')
    
if len(np.shape(radius)) > 0:
    Rs, spectrum = radius
    figure()
    plot(Rs*1e9, spectrum, 'x-')
    plt.xlabel("Radius [nm]")
    plt.ylabel("Coefficient")
    if outpath is not None:   
        plt.savefig(base_name+'_rSpectrum_fig.pdf')
    figure()
    plt.title('LSE = {:.4e}, pixel = {:.3f} um'.format(
        lse, pixel_size*1e6))   
else:
    figure()
    plt.title('r= {:.2f} nm, LSE = {:.4e}, pixel = {:.3f} um'.format(
        radius*1e9, lse, pixel_size*1e6))   
#==============================================================================
# Plot
#==============================================================================

X=np.arange(len(dp.get_fax(profiles)))*pixel_size*1e6

plot(X,dp.get_fax(profiles))
plot(X,dp.get_fax(fits))

plt.xlabel('Position [$\mu$m]')
plt.ylabel('Normalised amplitude')

#==============================================================================
# Save
#%%============================================================================

if outpath is not None:
    
    plt.savefig(base_name+'_fig.pdf') 
    with open(base_name+'_result.txt','wb') as f:
        f.write("LSE: {:e}\n".format(lse).encode())
        f.write("Apparent pixel size: {:f} um\n".format(pixel_size*1e6).encode())
        if len(np.shape(radius))>0:
            f.write("Radii:\n".encode())
            np.savetxt(f,radius[0])
            f.write("Spectrum:\n".encode())
            np.savetxt(f,radius[1])
            
        else:
            f.write("Radius: {:f} nm".format(radius*1e9).encode())
        f.write("Profiles:".encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        
