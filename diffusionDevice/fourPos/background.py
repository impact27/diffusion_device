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
from scipy.ndimage.morphology import binary_erosion

_umChannelWidth=100


def channels_edges(bg, pixs,angle=None,std=10e-6):
    """
    Get the position of the edges
    
    Parameters
    ----------
    bg:  2d array
        image containning the 4 channels 
    pixs: float
        The pixel size in [m]
    angle: float
        if given, angle at which the edges are 
    std: float
        Tolerence on wall position in [m]
        
    Returns
    -------
    edges: 1d integer array
        Position in the rotated image of the edges in pixels
    
    """
    
    bg=bg/rmbg.polyfit2d(bg)
    if angle is not None:
        bg=ir.rotate_scale(bg,-angle,1,borderValue=np.nan)
        

    prof=gfilter(np.nanmean(bg,0),3)
    edges=np.abs(np.diff(prof))
    edges[np.isnan(edges)]=0
    #create approximate walls
    x=np.arange(len(edges))*pixs
    gwalls=np.zeros(len(edges),dtype=float)
    for center in np.arange(1,9)*_umChannelWidth*1e-6:
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
    
    '''
    from matplotlib.pyplot import plot, figure, imshow
    figure()
    imshow(bg)
    figure()
    plot(edges)
    plot(gwalls)
    #'''
    #Get the positions
    edges=np.squeeze(msr.maximum_position(edges,label,range(1,n+1)))
    assert(len(edges)==8)
    return edges

def channels_mask(bg, pixs, angle=None, edgesOut=None):
    """
    Get the mask from the image
    
    Parameters
    ----------
    bg:  2d array
        image containning the 4 channels 
    pixs: float
        The pixel size in [m]
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
    maxpos=channels_edges(bg,pixs,angle)
    if edgesOut is not None:
        edgesOut[:]=maxpos
    #Fill mask
    mask=np.ones(bg.shape)
    for i in range(len(maxpos)//2):
        mask[:,maxpos[2*i]:maxpos[2*i+1]]=0
    return mask

def bg_angle(im,bg,infoDict=None):
    """
    get the angle by remove_curve_background
    
    Parameters
    ----------
    im:  2d array
        image containning the 4 channels
    bg:  2d array
        background 
    infoDict: dict
        infos out
        
    Returns
    -------
    angle: float
        the image orientation angle
    """
    tmpout=rmbg.remove_curve_background(im,bg,infoDict=infoDict,bgCoord=True)
    return dp.image_angle(tmpout)

def remove_bg(im,bg, pixs,edgesOut=None):
    """
    Flatten and background subtract images
    
    Parameters
    ----------
    im:  2d array
        list of images containning the 4 channels 
    bg: 2d array
        Background corresponding to the list
    pixs: float
        The pixel size in [m]
    edgesOut: 1d array
        output for the edges
        
    Returns
    -------
    flatIm: 2d array
        Flattened image
    
    """
    #Get bg angle (the other images are the same)
    infoDict={}
    angle=bg_angle(im,bg,infoDict)
    #Get the mask
    maskbg=channels_mask(bg,pixs,angle,edgesOut)
    #rotate and flatten the bg
    bg=ir.rotate_scale(bg,-angle,1,borderValue=np.nan)
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    
    maskim=ir.rotate_scale_shift(maskbg, infoDict['diffAngle'],
                                         infoDict['diffScale'],
                                         infoDict['offset'], 
                                         borderValue=np.nan)>.5   
                                 
    maskim=binary_erosion(maskim,iterations=15)
    #Get Intensity
    ret=rmbg.remove_curve_background(im,bg,maskbg=maskbg,maskim=maskim,
                                     bgCoord=True,reflatten=True)
    return ret

def extract_profiles(im,bg,pixs):
    """
    Extract diffusion profiles
    
    
    Parameters
    ----------
    im:  2d array
        image containning the 4 channels 
    bg: 2d array
        Background image
    pixs: float
        The pixel size in [m]
        
    Returns
    -------
    profiles: 2d array
        list of profiles
    """
    #get edges 
    edges=np.empty(8,dtype=int)
    #Get flattened image
    flat_im=remove_bg(im,bg,pixs,edges)
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
        
    """
    from matplotlib.pyplot import plot, figure, imshow
    figure()
    imshow(flat_im)
    figure()
    plot(imProf)
    #"""
    
    return profiles

def apparent_pixel_size(bg,estimated_pix_size,im=None):
    """
    Compute the apparent pixel size
    
    Parameters
    ----------
    bg: 2d array
        Background image
    estimated_pix_size: float
        The estimated pixel size in [m]
        
    Returns
    -------
    pixsize: float
        The apparent pixel size
    """
    if im is None:
        a=ir.orientation_angle(bg)-np.pi/2
    else:
        a=bg_angle(im,bg)
    edges=channels_edges(bg,estimated_pix_size,a)
    #2000 is (channel width + gap ) *10
    return np.mean([20*_umChannelWidth/np.mean(np.diff(edges[::2])),
                    20*_umChannelWidth/np.mean(np.diff(edges[1::2]))])