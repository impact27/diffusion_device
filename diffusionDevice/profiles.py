# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:25:47 2017

@author: quentinpeter
"""
import numpy as np
from diffusionDevice.basisgenerate import getprofiles

def fit_monodisperse_radius(profiles, flowRate, pixs, readingpos,
                                Wz=50e-6,
                                Zgrid=11,
                                ignore=10e-6,
                                Rs=np.arange(.5,10,.5)*1e-9,):
    """
    Find the best monodisperse radius
    
     Parameters
    ----------
    profiles: 1d array
        the set of profiles
    flowRate: float
    Wz=50e-6
    Zgrid=11
    ignore=10
    pixs=.847e-6
    Rs=np.arange(.5,10,.5)*1e-6
    readingpos 
        
    Returns
    -------
    radii: float
        The best radius fit
    """
    
    ignore=int(ignore/pixs)
    
    #Get basis function    
    Wy=pixs*np.shape(profiles)[1]
    Basis=getprofiles(profiles[0],flowRate,Rs,Wy=Wy,Wz=Wz,
                      Zgrid=Zgrid,readingpos=readingpos)
    
#    for basisrad in Basis:
#        for b in basisrad:
#            b*=np.mean(profiles[0])/np.mean(b)
    #Compute residues
    p=profiles[1:]
    res=np.empty(len(Rs),dtype=float)
    for i,b in enumerate(Basis):
        res[i]=np.sqrt(np.mean(np.square(b-p)[:,ignore:-ignore]))

    '''
    from matplotlib.pyplot import figure, plot
    figure()
    plot(Rs,res)
    #'''
    
    #Use linear combination between the two smallest results
    i,j=np.argsort(res)[:2]
    b1=Basis[i,:,ignore:-ignore]
    b2=Basis[j,:,ignore:-ignore]
    p0=p[:,ignore:-ignore]
    c=-np.sum((b1-b2)*(b2-p0))/np.sum((b1-b2)**2)
    
    #Get resulting r
    r=c*(Rs[i]-Rs[j])+Rs[j]
    return r

def center(prof):
    """
    Uses correlation between Y and the mirror image of Y to get the center
    
    Parameters
    ----------
    prof:  1d array
        Profile 
        
    Returns
    -------
    center: float
        The center position in pixel units
    
    """
    
    #We must now detect the position of the center. We use correlation
    #Correlation is equivalent to least squares (A-B)^2=-2AB+ some constants
    prof=prof.copy()
    prof[np.isnan(prof)]=0
    Yi=prof[::-1]
    corr=np.correlate(prof,Yi, mode='full')
    Y=corr
    X=np.arange(len(Y))
    args=np.argsort(Y)
    x=X[args[-7:]]
    y=Y[args[-7:]]
    coeffs=np.polyfit(x,np.log(y),2)
    center=-coeffs[1]/(2*coeffs[0])
    center=(center-(len(corr)-1)/2)/2+(len(prof)-1)/2
    return center

def baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile
    
    Parameters
    ----------
    prof:  1d array
        Profile 
    frac: float, defaults .05
        Fraction of the profile to use
        
    Returns
    -------
    baseline: 1d array
        The baseline
    
    """
    #we use 5% on left side to get the correct 0:
    #Get left and right zeros
    argvalid=np.argwhere(np.isfinite(prof))
    lims=np.squeeze([argvalid[0],argvalid[-1]])
    left=int(lims[0]+frac*np.diff(lims))
    right=int(lims[1]-frac*np.diff(lims))
    leftZero=np.nanmean(prof[lims[0]:left])
    rightZero=np.nanmean(prof[right:lims[1]])
        
    #Send profile to 0
    baseline=np.linspace(leftZero,rightZero,len(prof))
    return baseline

def flat_baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile
    
    Parameters
    ----------
    prof:  1d array
        Profile 
    frac: float, defaults .05
        Fraction of the profile to use
        
    Returns
    -------
    baseline: 1d array
        The flat baseline
    
    """
    #we use 5% on left side to get the correct 0:
    #Get left and right zeros
    leftZero=np.nanmean(prof[:int(frac*len(prof))])
    rightZero=np.nanmean(prof[-int(frac*len(prof)):])
        
    #Send profile to 0
    ret=np.zeros(prof.shape)+np.mean([leftZero,rightZero])
    return ret

def image_angle(image, maxAngle=np.pi/7):
    """
    Analyse an image with y invariance to extract a small angle.
    
    Parameters
    ----------
    image:  2d array
        image with y invariance 
    maxAngle: float, defaults np.pi/7
        Maximal rotation angle 
        
    Returns
    -------
    angle: float
        The rotation angle
    
    """
    #Difference left 50% with right 50%
    #We want to slice in two where we have data
    argvalid=np.argwhere(np.isfinite(np.nanmean(image,1)))
    lims=np.squeeze([argvalid[0],argvalid[-1]])
    #should we flatten this?
    top=np.nanmean(image[lims[0]:np.mean(lims,dtype=int)] ,0)
    bottom=np.nanmean(image[np.mean(lims,dtype=int):lims[1]],0)
    #Remouve nans
    top[np.isnan(top)]=0
    bottom[np.isnan(bottom)]=0
    #correlate
    C=np.correlate(top,bottom, mode='same')
    X=np.arctan((np.arange(len(C))-len(C)/2)/((lims[1]-lims[0])/2))
    valid=np.abs(X)<maxAngle
    x=X[valid]
    c=C[valid]         
    angle=x[c.argmax()]
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X,C)
    plt.plot([maxAngle,maxAngle],[-.1,.3])
    plt.plot([-maxAngle,-maxAngle],[-.1,.3])
    plt.plot([angle,angle],[0,.2])
    #"""
    if np.abs(angle) >maxAngle:
        angle=0
    return angle

def initprocess(profile, mode):
    if mode == 'none':
        return profile
    elif mode == 'gaussian' or mode == 'tails':
        Y=profile
        X=np.arange(len(Y))
        valid=Y>.5*Y.max()
        gauss=np.exp(np.poly1d(np.polyfit(X[valid],np.log(Y[valid]),2))(X))
        if mode=='gaussian':
            return gauss
        remove=gauss<.01*gauss.max()
        profile[remove]=0
        return profile
