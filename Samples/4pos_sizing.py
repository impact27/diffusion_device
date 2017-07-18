"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
import diffusion_device.profile as dp
import diffusion_device.four_channels_image as dd4
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
from diffusion_device.json import full_fit
#==============================================================================
# Settings
#==============================================================================

#File Name. If using UV, [image filename, background filename]
outpath = 'output'
settingsfn='../diffusion_device/tests/test_data/UVim300ulph_fitSettings.json'
settingsfn='../diffusion_device/tests/test_data/Brightim900ulph_fitSettings.json'

#==============================================================================
# Fit
#==============================================================================

radius, profiles, fits, lse, pixel_size, __ = full_fit(settingsfn)

#==============================================================================
# Plot
#==============================================================================

X=np.arange(len(dp.get_fax(profiles)))*pixel_size*1e6
figure()
plot(X,dp.get_fax(profiles))
plot(X,dp.get_fax(fits))
plt.title('r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
        radius*1e9, lse, pixel_size*1e6))   
plt.xlabel('Position [$\mu$m]')
plt.ylabel('Normalised amplitude')

#==============================================================================
# Save
#==============================================================================

if outpath is not None:
    import os
    import shutil
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    base_name = os.path.join(outpath, 
                             os.path.splitext(os.path.basename(settingsfn))[0])
    
    plt.savefig(base_name+'_fig.pdf')
    shutil.copy(settingsfn, base_name + '.json')     
    with open(base_name+'_result.txt','wb') as f:
        f.write("""Radius: {:f} nm
LSE: {:e}
Apparent pixel size: {:f} um
Profiles:
""".format(radius*1e9,lse,pixel_size*1e6).encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        
