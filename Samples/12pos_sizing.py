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

#File Name. If using UV, [image filename, background filename]
settingsfn='../diffusion_device/tests/test_data/350ulh_12pos/fitsettings.json'
save=False

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
#==============================================================================
# Plot
#==============================================================================

X=np.arange(len(dp.get_fax(profiles)))*pixel_size*1e6
figure()
plot(X,dp.get_fax(profiles))
plot(X,dp.get_fax(fits))
plt.title('r= {:.2f} nm, LSE = {:.2e}'.format(
        radius*1e9, lse))   
plt.xlabel('Position [$\mu$m]')
plt.ylabel('Normalised amplitude')

#==============================================================================
# Save
#==============================================================================

if save:
    from os import path
    name=path.splitext(settingsfn)[0]
    plt.savefig(name+'_fig.pdf')
            
    with open(name+'_result.txt','wb') as f:
        f.write("""Radius: {:f} nm
LSE: {:e}
Profiles:
""".format(radius*1e9,lse).encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        
