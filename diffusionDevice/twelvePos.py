# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:26:20 2017

@author: quentinpeter
"""
import numpy as np
import background_rm as rmbg
import image_registration.image as ir
import diffusionDevice.profiles as dp
import warnings
warnings.filterwarnings('ignore', 'Mean of empty slice',RuntimeWarning)

def extract_profile(im,bg, chanWidth=300e-6, pixsize=2*8.47e-6/10,
               outim=None):
    """
    Extract profile from image
    
    Parameters
    ----------
    im: 2d array
        image 
    bg: 2d array
        background
    chanWidth: float, defaults 300e-6
        channel width  in [m]
    pixsize: float, defaults 2*8.47e-6/10
        pixel size in [m]
    widthcut: integer,0
        number of pixel to cut from the side (TO REMOVE) 
    outim: 2d array, defaults None
        Get processed image if needed
        
    Returns
    -------
    X: 1d array
        X position of the profile
    prof: 1d array
        The profile
    
    """
    im=np.asarray(im,dtype=float)
    bg=np.asarray(bg,dtype=float)
    #remove dust peaks on images
    bg[rmbg.getPeaks(bg, maxsize=50*50)]=np.nan
    im[rmbg.getPeaks(im, maxsize=50*50)]=np.nan   
       
    #Get the X positions (perpendicular to alignent axis) and check wide enough
    X=np.arange(im.shape[0])*pixsize
    assert(1.2*chanWidth<X[-1])
    
    #Get the approximate expected channel position
    channel=np.absolute(X-X[-1]/2)<.6*chanWidth
    
    #Create mask to ignore channel when flattening image
    mask=np.ones(im.shape,dtype=bool)
    mask[channel,:]=False
    
    #Get data
    output=rmbg.remove_curve_background(im,bg, method='mask',mask=mask)
        
    output=ir.rotate_scale(output,dp.image_angle(output)
                            ,1, borderValue=np.nan)
    prof=np.nanmean(output,1)
    
    #Center X
    center=dp.center(prof)*pixsize
    inchannel=np.abs(X-center)<.45*chanWidth
    X=X-(dp.center(prof[inchannel])+np.argmax(inchannel))*pixsize
    
    #get what is out
    out=np.logical_and(np.abs(X)>.55*chanWidth,np.isfinite(prof))
    #fit ignoring extreme 10 pix
    fit=np.polyfit(X[out][10:-10],prof[out][10:-10],2)
    bgfit=fit[0]*X**2+fit[1]*X+fit[2]
    #Flatten the profile
    prof=(prof+1)/(bgfit+1)-1
    if outim is not None:
        outim[:]=output[:]
    #We restrict the profile to channel width - widthcut
    Npix=int(chanWidth//pixsize)
    c=np.argmin(np.abs(X))
    print(Npix,c)
    assert(Npix//2<c)
    assert(len(prof)-c>Npix//2)
    channel=slice(c-Npix//2,c+Npix//2+Npix%2)
    
    prof=prof[channel]
    X=X[channel]
    return X, prof
 
def defaultReadingPos():
    '''
    Get the default reading positions for the 12 points diffusion device
    '''
    return np.array([3.5,
                     5.3,
                     8.6,
                     10.3,
                     18.6,
                     20.4,
                     28.6,
                     30.4,
                     58.7,
                     60.5,
                     88.7,
                     90.5])*1e-3    
    
    