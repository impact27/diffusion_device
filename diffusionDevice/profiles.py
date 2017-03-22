# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:25:47 2017

@author: quentinpeter
"""
import numpy as np
from diffusionDevice.basisgenerate import getprofiles

def fit_monodisperse_radius(profiles,flowRate,Wz=50e-6,
               Zgrid=11,
               ignore=10,
               pixs=.847e-6,
               Rs=np.arange(.5,10,.5)*1e-6,
               readingpos=np.array([ 0.004183,  0.021446,  0.055879])):
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
    #Get basis function
    initprof=profiles[0].copy()
    
    #Remove noise on side
    x=np.arange(len(initprof))
    valid=initprof>.5*initprof.max()
    Y=np.log(initprof[valid])
    X=x[valid]
    F=np.poly1d(np.polyfit(X,Y,2))
    fit=np.exp(F(x))
    remove=fit<.01*fit.max()
    initprof[remove]=0

#    from matplotlib.pyplot import figure, plot
#    figure(0)
#    plot(initprof/initprof.sum())
    
    Wy=pixs*len(initprof)
    Basis=getprofiles(initprof,flowRate,Rs,Wy=Wy,Wz=Wz,
                      Zgrid=Zgrid,readingpos=readingpos)
    #Compute residues
    p=profiles[1:]
    res=np.empty(len(Rs),dtype=float)
    for i,b in enumerate(Basis):
        res[i]=np.sqrt(np.mean(np.square(b-p)[:,ignore:-ignore]))

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
    center=np.correlate(prof,Yi, mode='same').argmax()/2+len(prof)/4
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
    Analyse an image with x invariance to extract a small angle.
    
    Parameters
    ----------
    image:  2d array
        image with x invariance 
    maxAngle: float, defaults np.pi/7
        Maximal rotation angle 
        
    Returns
    -------
    angle: float
        The rotation angle
    
    """
    #Difference left 50% with right 50%
    #We want to slice in two where we have data
    argvalid=np.argwhere(np.isfinite(np.nanmean(image,0)))
    lims=np.squeeze([argvalid[0],argvalid[-1]])
    #should we flatten this?
    left=np.nanmean(image[:,lims[0]:np.mean(lims,dtype=int)] ,1)
    right=np.nanmean(image[:,np.mean(lims,dtype=int):lims[1]],1)
    #Remouve nans
    left[np.isnan(left)]=0
    right[np.isnan(right)]=0
    #correlate
    C=np.correlate(left,right, mode='same')
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

