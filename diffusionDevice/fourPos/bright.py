# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:18:33 2017

@author: quentinpeter
"""
import numpy as np
import scipy.ndimage
gfilter=scipy.ndimage.filters.gaussian_filter1d
from scipy.ndimage.filters import maximum_filter1d
import diffusionDevice.profiles as dp
import background_rm as rmbg
import image_registration.image as ir
from scipy import interpolate


def image_infos(im):
    """
    Get the image angle, channel width, proteind offset, and origin
    
    Parameters
    ----------
    im: 2d array
        The image
        
    Returns
    -------
    dict: dictionnary
        dictionnary containing infos
    
    """
    #Detect Angle
    angle=dp.image_angle(im-np.median(im))
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    #Get channels infos
    w,a,origin=straight_image_infos(im)
    
    retdict={
            'angle':angle,
            'origin':origin,
            'width':w,
            'offset':a}
    return retdict

def straight_image_infos(im):
    """
    Get the channel width, proteind offset, and origin from a straight image
    
    Parameters
    ----------
    im: 2d array
        The image
        
    Returns
    -------
    w: float
        Channel width in pixels
    a: float
        offset of the proteins in the channel
    origin: float
        Position of the first channel center
    
    """
    profiles=np.nanmean(im,0)
    
    #Find max positions
    fprof=gfilter(profiles,3)
    fprof=profiles
    maxs=np.where(maximum_filter1d(fprof,100)==fprof)[0]
    assert(len(maxs)==4)    
    maxs=np.asarray(maxs,dtype=float)
    for i,amax in enumerate(maxs):
        amax=int(amax)
        y=np.log(profiles[amax-10:amax+10])
        x=np.arange(len(y))
        coeff=np.polyfit(x,y,2)
        maxs[i]=-coeff[1]/(2*coeff[0])-10+amax
        
    #Deduce relevant parameters
    w=(maxs[2]-maxs[0])/4
    a=w+(maxs[0]-maxs[1])/2
    origin=maxs[0]-a
    
    return w,a,origin
    

def flat_image(im,frac=.7,infosOut=None):
    """
    Flatten input images
    
    Parameters
    ----------
    im: 2d array
        The image
    frac: float
        fraction of the profile taken by fluorescence from channels
    infosOut: dict, defaults None
        dictionnary containing the return value of straight_image_infos
        
    Returns
    -------
    im: 2d array
        The flattened image
    
    """
    #Detect Angle
    angle=dp.image_angle(im-np.median(im))
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    #Get channels infos
    w,a,origin=straight_image_infos(im)
    #get mask
    mask=np.ones(np.shape(im)[1])
    for i in range(4):
        amin=origin+2*i*w-frac*w
        amax=origin+2*i*w+frac*w
        mask[int(amin):int(amax)]=0
    mask=mask>0
    mask=np.tile(mask[None,:],(np.shape(im)[0],1))
    #Flatten
    im=im/rmbg.polyfit2d(im,mask=mask)-1
    if infosOut is not None:
        infosOut['infos']=(w,a,origin)
    return im
    
def extract_profiles_flatim(im,infos):
    '''
    Extract profiles from flat image
    
    Parameters
    ----------
    im: 2d array
        The flat image
    infos: dict
        dictionnary containing the return value of straight_image_infos
        
    Returns
    -------
    profiles: 2d array
        The four profiles
    '''
    #Find positions
    w,a,origin=infos
    image_profile=np.nanmean(im,0)
    
    #Extract one by one
    Npix=int(np.round(w))
    profiles=np.zeros((4,Npix))
    
    for i in range(4):   
        X=np.arange(len(image_profile))-(origin+2*i*w)        
        Xc=np.arange(Npix)-(Npix-1)/2
        finterp=interpolate.interp1d(X, image_profile)
        protoprof = finterp(Xc)
        #switch if uneven
        if i%2==1:
            protoprof=protoprof[::-1]
            
        profiles[i]=protoprof
    
    #If image upside down, turn
    if profiles[-1].max()>profiles[0].max():
        profiles=profiles[::-1] 
        
    """
    from matplotlib.pyplot import plot, figure, imshow
    figure()
    imshow(im)
    figure()
    plot(image_profile)
    #"""
    return profiles

def extract_profiles(im):
    '''
    Extract profiles from image
    
    Parameters
    ----------
    im: 2d array
        The flat image
        
    Returns
    -------
    profiles: 2d array
        The four profiles
    '''
    infos={}
    im=flat_image(im,infosOut=infos)
    profiles=extract_profiles_flatim(im,infos['infos'])
    return profiles