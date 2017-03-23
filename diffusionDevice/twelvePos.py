# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:26:20 2017

@author: quentinpeter
"""
import numpy as np
import background_rm as rmbg
import image_registration.image as ir
import image_registration.channel as cr
import diffusionDevice.profiles as dp
import scipy
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
    
def outChannelMask(im, chAngle=0):
    """Creates a mask that excludes the channel
    
    Parameters
    ----------
    im: 2d array
        The image
    chAngle: number
        The angle of the channel in radians
    
    Returns
    -------
    mask: 2d array
        the mask excluding the channel
        
    Notes
    -----
    The channel should be clear(ish) on the image. 
    The angle should be aligned with the channel
    

    """
    im=np.array(im,dtype='float32')
    #Remove clear dust
    mask=rmbg.backgroundMask(im, nstd=6)
    im[~mask]=np.nan
    
    #get edge
    scharr=cr.Scharr_edge(im)
    #Orientate image along x if not done
    if chAngle !=0:
        scharr= ir.rotate_scale(scharr, -chAngle,1,np.nan)
        
    #get profile
    prof=np.nanmean(scharr,1)
    #get threshold
    threshold=np.nanmean(prof)+3*np.nanstd(prof)
    mprof=prof>threshold
    edgeargs=np.flatnonzero(mprof)
    
    if edgeargs.size > 2:
        mask=np.zeros(im.shape)
        mask[edgeargs[0]-5:edgeargs[-1]+5,:]=2
        if chAngle !=0:
            mask= ir.rotate_scale(mask, chAngle,1,np.nan)
        mask=np.logical_and(mask<1, np.isfinite(im))
    else:
        mask= None
    return mask
    
def outGaussianBeamMask(data, chAngle=0):
    """
    A single, straight, protein beam is present. It is "Sinking" the profile 
    such as the sides are leaning toward the center
    """
    data=np.asarray(data)
    
    #Filter to be used
    gfilter=scipy.ndimage.filters.gaussian_filter1d
    
    #get profile
    if chAngle!=0:
        data=ir.rotate_scale(data, -chAngle,1,np.nan)
    profile=np.nanmean(data,1)
    
    #guess position of max
    amax= profile.size//2
    
    #get X and Y
    X0=np.arange(profile.size)-amax
    Y0=profile
    
    #The cutting values are when the profiles goes below zero
    rlim=np.flatnonzero(np.logical_and(Y0<0,X0>0))[0]
    llim=np.flatnonzero(np.logical_and(Y0<0,X0<0))[-1]
    
    #We can now detect the true center
    fil=gfilter(profile,21)
    X0=X0-X0[np.nanargmax(fil[llim:rlim])]-llim
    
    #restrict to the correct limits
    X=X0[llim:rlim]
    Y=Y0[llim:rlim]-np.nanmin(Y0)
    
    #Fit the log, which should be a parabola
    c=np.polyfit(X,np.log(Y),2)
    
    #Deduce the variance
    var=-1/(2*c[0])
    
    #compute the limits (3std, restricted to half the image)
    mean=np.nanargmax(fil[llim:rlim])+llim
    dist=int(3*np.sqrt(var))
    if dist > profile.size//4:
        dist = profile.size//4
    llim=mean-dist
    if llim < 0:
        return None
    rlim=mean+dist
    if rlim>profile.size:
        return None
    
    #get mask
    mask=np.ones(data.shape)
    
    if chAngle!=0:
        idx=np.indices(mask.shape)
        
        
        idx[1]-=mask.shape[1]//2
        idx[0]-=mask.shape[0]//2
        X=np.cos(chAngle)*idx[1]+np.sin(chAngle)*idx[0]
        Y=np.cos(chAngle)*idx[0]-np.sin(chAngle)*idx[1]
        
        mask[np.abs(Y-mean+mask.shape[0]//2)<dist]=0
        
    else:    
        mask[llim:rlim,:]=0
    
    #mask=np.logical_and(mask>.5, np.isfinite(data))
    mask=mask>.5
    return mask
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    #plot profile and fit
    valmax=np.nanmax(Y)
    plt.plot(X0,Y0)
    plt.plot(X0,valmax*np.exp(-(X0**2)/(2*var))+np.nanmin(Y0))
    plt.plot([llim-mean,llim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    plt.plot([rlim-mean,rlim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    #"""
    