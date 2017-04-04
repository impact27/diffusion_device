# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:21:01 2017

@author: quentinpeter
"""
import numpy as np
import matplotlib.image as mpimg
from . import bright, background
import diffusionDevice.profiles as dp
import diffusionDevice.basisgenerate as ddbg

def defaultReadingPos():
    '''
    Get the default reading positions for the 4 points diffusion device
    
    Returns
    -------
    readingPos: 1d array
        The reading positions
    '''
    return np.array([  4183, 21446, 55879])*1e-6

def size_image(im,Q,Wz,pixsize,readingpos=None,Rs=None,chanWidth=100e-6,*,
                Zgrid=11,ignore=5e-6,normalize_profiles=True,initmode='none',
                data_dict=None):
    
    """
    Get the hydrodynamic radius from the images
    
    Parameters
    ----------
    images: 2d image or file name OR 2x 2d images
        If this is a string, it will be treated like a path
        If one image, treated like regular fluorescence image
        If two images, treated like image and background
    Q: float
        Flow rate in [ul/h]
    Wz: float
        Height of the channel in [m]
    pixsize: float
        Pixel size in [m]
    readingpos: 1d float array, defaults None
        Position at which the images are taken. If None, take the defaults
    Rs: 1d array, defaults None
        Hydrodimamic radii to simulate in [m].
        If None: between .5 and 10 nm
    chanWidth: float, default 100e-6
        The channel width in [m]
    Zgrid: int, defaults 11
        Number of Z slices
    ignore: float, defaults 5e-6
        Distance to sides to ignore
    normalize_profiles: Bool, defaults True
        Should the profiles be normalized?
    initmode: str, defaults 'none'
        The processing mode for the initial profile (See profiles.py)
    data_dict: dict, defaults None
        Output to get the profiles and fits
        
    Returns
    -------
    r: float
        Radius in [m]
    
    """
    
    #Check images is numpy array
    im=np.asarray(im)
    
    #Fill missing arguments
    if readingpos is None:
        readingpos=defaultReadingPos()
    if Rs is None:
        Rs=np.arange(.5,10,.5)*1e-9
    
    #load images if string
    if im.dtype.type==np.str_:
        if len(np.shape(im))==0:
            im=mpimg.imread(str(im))
        elif len(np.shape(im))==1:
            im=np.asarray([mpimg.imread(fn) for fn in im])
    #get profiles
    if len(np.shape(im))==2:
        #Single image
        profiles=bright.extract_profiles(im)
    elif len(np.shape(im))==3 and np.shape(im)[0]==2:
        #images and background
        profiles= background.extract_profiles(im[0],im[1])
        
    #normalize if needed
    if normalize_profiles:
        for p in profiles:
            p/=np.sum(p)
    
    #treat init profile
    profiles[0]=dp.initprocess(profiles[0],initmode)

    #Get best fit
    r=dp.fit_monodisperse_radius(profiles,flowRate=Q,Wz=Wz,
                   Zgrid=Zgrid,
                   ignore=ignore,
                   pixs=pixsize,
                   Rs=Rs,
                   readingpos=readingpos)
    
    #fill data if needed
    if data_dict is not None:
        data_dict['profiles']=profiles
        data_dict['fits']=ddbg.getprofiles(profiles[0],Q=Q, Rs=[r], 
                             Wy = len(profiles[0])*pixsize, Wz= Wz, Zgrid=Zgrid,
                             readingpos=readingpos)[0]
    return r