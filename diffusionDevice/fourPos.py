# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:26:48 2017

@author: quentinpeter
"""
import image_registration.image as ir
import image_registration.channel as cr
import numpy as np
from scipy.ndimage.morphology import binary_dilation as dilate
import scipy.ndimage.measurements as msr
import background_rm as rmbg
import diffusionDevice.profiles as dp
import scipy
gfilter=scipy.ndimage.filters.gaussian_filter1d


def channels_edges(im, angle=None,pixs=.847,std=10):
    """
    Get the position of the edges
    
    Parameters
    ----------
    im:  2d array
        image containning the 4 channels 
    angle: float
        if given, angle at which the edges are 
        
    Returns
    -------
    edges: 1d integer array
        Position in the rotated image of the edges in pixels
    
    """
    
    im=im/rmbg.polyfit2d(im)
    if angle is not None:
        im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
        

    prof=gfilter(np.nanmean(im,0),3)
    edges=np.abs(np.diff(prof))
    edges[np.isnan(edges)]=0
    #create approximate walls
    x=np.arange(len(edges))*pixs
    gwalls=np.zeros(len(edges),dtype=float)
    for center in np.arange(100,801,100):
        gwalls+=edges.max()*np.exp(-(x-center)**2/(2*std**2))
    #Get best fit for approximate walls
    c=int(np.correlate(edges,gwalls,mode='same').argmax()-len(gwalls)/2)
    #Roll
    gwalls=np.roll(gwalls,c)
    if c<0:
        gwalls[c:]=0
    else:
        gwalls[:c]=0
    #label wall position
    label,n=msr.label(gwalls>.1*gwalls.max())
    
    #'''
    from matplotlib.pyplot import plot, figure
    figure()
    plot(edges)
    plot(gwalls)
    #'''
    #Get the positions
    edges=np.squeeze(msr.maximum_position(edges,label,range(1,n+1)))
    return edges

def channels_mask(im, angle=None, edgesOut=None):
    """
    Get the mask from the image
    
    Parameters
    ----------
    im:  2d array
        image containning the 4 channels 
    angle: float
        if given, angle at which the edges are 
    edgesOut: 1d array
        output for the edges
        
    Returns
    -------
    mask: 2d array
        mask marking the channels 
    
    """
    #Find edges position
    maxpos=channels_edges(im,angle)
    if edgesOut is not None:
        edgesOut[:]=maxpos
    #Fill mask
    mask=np.ones(im.shape)
    for i in range(len(maxpos)//2):
        mask[:,maxpos[2*i]:maxpos[2*i+1]]=0
    return mask

def remove_bg(im,bg,edgesOut=None,bgIOut=None):
    """
    Flatten and background subtract images
    
    Parameters
    ----------
    im:  2d array
        list of images containning the 4 channels 
    bg: 2d array
        Background corresponding to the list
    edgesOut: 1d array
        output for the edges
    bgIOut: 2d array
        output for the flattened background image
        
    Returns
    -------
    flatIms: 3d array
        list of flattened images 
    
    """
    #Get bg angle (the other images are the same)
    tmpout=rmbg.remove_curve_background(im,bg)
    angle=-dp.image_angle(np.rot90(tmpout))
    #Get the mask
    mask=channels_mask(bg,angle,edgesOut)
    #rotate and flatten the bg
    bg=ir.rotate_scale(bg,-angle,1,borderValue=np.nan)
    if bgIOut is None:
        bgIOut=np.empty(bg.shape,dtype=float)
    bgIOut[:]=bg/rmbg.polyfit2d(bg,mask=mask)
    #rotate
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    #Get Intensity
    ret=rmbg.remove_curve_background(im,bg,mask=mask)
    return ret

def extract_profiles(im,bg):
    """
    Extract diffusion profiles
    
    
    Parameters
    ----------
    im:  2d array
        image containning the 4 channels 
    bg: 2d array
        Background image
        
    Returns
    -------
    profiles: 2d array
        list of profiles
    """
    #get edges 
    edges=np.empty(8,dtype=int)
    #Get flattened image
    flat_im=remove_bg(im,bg,edges)
    #Get channel width
    width=int(np.mean(np.diff(edges)[::2]))
    #Profile
    imProf=np.nanmean(flat_im,0)
    #Output profiles
    profiles=np.empty((4,width),dtype=float)
    #Extract profiles
    firstcenter=None
    for i,(e,prof) in enumerate(zip(edges[1::2]+edges[::2],profiles)):
        #e is 2*center of the channel
        amin=(e-width)//2
        amax=(e+width)//2
        p=imProf[amin:amax]
        #All even profiles are switched
        if i%2==1:
            p=p[::-1]
        #Align by detecting center
        c=dp.center(p)
        if firstcenter is not None:
            diff=c-firstcenter
            if i%2==1:
                diff*=-1
            diff=int(diff)
            p=imProf[amin+diff:amax+diff]
            if i%2==1:
                p=p[::-1]
        else:
            firstcenter=c
        prof[:]=p
    #If image is inverted
    if profiles[-1].max()>profiles[0].max():
        profiles=profiles[::-1]
    return profiles

def apparent_pixel_size(bg):
    """
    Compute the apparent pixel size
    
    Parameters
    ----------
    bg: 2d array
        Background image
        
    Returns
    -------
    pixsize: float
        The apparent pixel size
    """
    a=ir.orientation_angle(bg)-np.pi/2
    edges=channels_edges(bg,a)
    return np.mean([2000/np.mean(np.diff(edges[::2])),
                    2000/np.mean(np.diff(edges[1::2]))])
    
def defaultReadingPos():
    '''
    Get the default reading positions for the 4 points diffusion device
    '''
    return np.array([  4183, 21446, 55879])*1e-6