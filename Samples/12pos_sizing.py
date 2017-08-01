"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
import diffusion_device.profile as dp
from diffusion_device.json import full_fit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure


#==============================================================================
# Settings
#==============================================================================

#Settings file name
settingsfn='../diffusion_device/tests/test_data/350ulh_12pos/fitsettings_poly.json'

#path to save the results
outpath = 'output'

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

radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)

# =============================================================================
# Plot
# =============================================================================

if outpath is not None:
    import os
    import shutil
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    base_name = os.path.join(outpath,  
                             os.path.splitext(os.path.basename(settingsfn))[0])
    shutil.copy(settingsfn, base_name + '.json')
    
X=np.arange(len(dp.get_fax(profiles)))*pixel_size*1e6

if len(np.shape(radius))>0:
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

plot(X,dp.get_fax(profiles))
plot(X,dp.get_fax(fits))   
plt.xlabel('Position [$\mu$m]')
plt.ylabel('Normalised amplitude')

#==============================================================================
# Save
#==============================================================================

if outpath is not None:   
    plt.savefig(base_name+'_fig.pdf')
            
    with open(base_name+'_result.txt','wb') as f:
        if len(np.shape(radius))>0:
            f.write("Radii:\n".encode())
            np.savetxt(f,radius[0])
            f.write("Spectrum:\n".encode())
            np.savetxt(f,radius[1])
        else:
            f.write("Radius: {:f} nm\n".format(radius*1e9).encode())
        
        f.write("LSE: {:e}\n".format(lse).encode())
        f.write("Profiles:\n".encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        
