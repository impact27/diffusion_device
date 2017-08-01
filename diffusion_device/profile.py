# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:25:47 2017

@author: quentinpeter
"""
import numpy as np
from .basis_generate import getprofiles
import scipy
gfilter = scipy.ndimage.filters.gaussian_filter1d
import warnings
from scipy.optimize import basinhopping, minimize
from itertools import combinations

def size_profiles(profiles, Q, Wz, pixsize, readingpos, Rs, *,
                  initmode='none', normalize_profiles=True, Zgrid=11,
                  ignore=10e-6, data_dict=None,
                  central_profile=False, nspecies=1):
    """Size the profiles
    
     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    Q: float
        Speed of the flow in [ul/h]
    Wz: float
        The channel height in [m]
    pixsize:float
        The pixel size in [m]
    readingpos: 1d float array
        The reading position of the profiles
    Rs: 1d float
        The test radii [m] 
    initmode: string, default 'none'
        How should the initial profile be processed 
        See init_process
    normalize_profiles: Bool, default True
        Should normalize profiles?
    Zgrid: integer, default 11
        Number of Z slices
    ignore: float, default 10e-6
        Ignore on the sides [m]
    data_dict: dictionnary
        If not None, returns infos
    central_profile: Bool, default False
        Should use central profile?
    nspecies: int, default 1
        number of species to fit
      
    Returns
    -------
    if nspecies==1:
        radii: float
            The best radius fit
    else:
        Rs, spectrum, the radii and corresponding spectrum
    """
    
    #convert ignore to px
    ignore = int(ignore/pixsize)
    
    if ignore == 0:
        pslice = slice(None)
    else:
        pslice = slice(ignore, -ignore)
    
    # Check input are arrays
    readingpos = np.asarray(readingpos)
    profiles = np.asarray(profiles)
    
    #normalize if needed
    if normalize_profiles:
        #if profile is mainly negative, error
        if np.any(np.sum((profiles*(profiles>0))[:, pslice],1) < 
                  5*-np.sum((profiles*(profiles<0))[:, pslice],1)):
            warnings.warn("Negative profile", RuntimeWarning)
        profiles/=np.sum(profiles[:, pslice],-1)[:,np.newaxis]
        
    #treat init profile
    init = init_process(profiles[0],initmode, ignore)
    #First reading pos is initial profile
    readingposfit = readingpos[1:]-readingpos[0]
    
    #Get basis function    
    Wy = pixsize*len(init)
    Basis = getprofiles(init, Q, Rs, Wy=Wy, Wz=Wz,
                      Zgrid=Zgrid, readingpos=readingposfit,
                      central_profile=central_profile) 
    
    
    
    if nspecies == 1:
        #Get best fit
        r = fit_radius(profiles[1:], Basis, Rs, ignore, nspecies=1)
        
        #fill data if needed
        if data_dict is not None:
            data_dict['initprof'] = init
            data_dict['fits'] = getprofiles(init,Q=Q, Radii=[r],
                                            Wy=Wy, Wz=Wz, Zgrid=Zgrid,
                                            readingpos=readingposfit,
                                            central_profile=central_profile)[0]
            
        return r
    else:
        spectrum = fit_radius(profiles[1:], Basis, Rs, ignore, nspecies=nspecies)
        
         #fill data if needed
        if data_dict is not None:
            data_dict['initprof'] = init
            data_dict['fits'] = np.sum(spectrum[:, np.newaxis, np.newaxis]
                                        * Basis, axis = 0)
        
        return Rs, spectrum
    


def fit_radius(profiles, Basis, Rs=None, ignore=0, nspecies=1):
    """Find the best monodisperse radius
    
     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    Basis: 3d array
        List of basis to fit. The first dimention must correspond to Rs
    Rs: 1d float
        The test radii [m] 
    ignore: int, default 0
        Ignore on the sides [px]
    nspecies: int
        Number of species to fit. 0=all.

    
    Returns
    -------
    spectrum: 
        The factors of Rs to get the best fit
    IF nspecies == 1:
    Radii: [m]
        The best radius fit
    """
    #How many pixels should we ignore?
    if ignore == 0:
        profslice = slice(None)
    else:
        profslice = slice(ignore, -ignore)
        
        
    Nb = len(Basis)
    flatbasis = np.reshape(Basis[:, :, profslice], (Nb, -1))
    flatprofs = np.ravel(profiles[:, profslice])
    M = np.zeros((Nb, Nb))
    b = np.zeros((Nb))
    
    psquare = np.sum(flatprofs*flatprofs)
    for i in range(Nb):
        b[i] = np.sum(flatbasis[i]*flatprofs)
        for j in range(Nb):
            M[i, j] = np.sum(flatbasis[i]*flatbasis[j])
            
    if nspecies == 1:
        return fit_monodisperse_radius(M, b, psquare, Rs)
    
    elif nspecies > 1:
        return fit_N_radius(M, b, psquare, nspecies)
        
    elif nspecies == 0:
        return fit_polydisperse_radius(M, b, psquare)
    
    else:
        raise RuntimeError('Number of species negative!')
    

def fit_monodisperse_radius(M, b, psquare, Rs=None):
    """Find the best monodisperse radius
    
    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    Rs: 1d float
        The test radii [m] 
    
    Returns
    -------
    radii: float
        The best radius fit
    """
    #get best residu
    res = psquare + np.diag(M)-2*b
    
    if Rs is None:
        ret = np.zeros_like(b)
        ret[np.argmin(res)]=1
        return ret
    
    i,j = np.argsort(res)[:2]
    #np.sum((b1-b2)*(p0-b2))/np.sum((b1-b2)**2)
    c = (b[i] - b[j] - M[i,j] + M[j,j])/(M[i,i] + M[j,j] - M[i,j] - M[j,i])
    
    #Get resulting r
    r = c*(Rs[i]-Rs[j])+Rs[j]
    
    '''
    from matplotlib.pyplot import figure, plot, title
    figure()
    plot(Rs,res)
    #'''
    
    if r < np.min(Rs):
        raise RuntimeError('The test radius are too big!')
    if r > np.max(Rs):
        raise RuntimeError('The test radius are too small!')
    
    return r

def fun(C, M, b, psquare):
    return psquare + C@M@C -2*C@b

def jac(C, M, b, psquare):
    return 2*C@M -2*b

def hess(C, M, b, psquare):
    return 2*M

def getconstr(Nb):
    constr = []
    
    # Need C[i]>0
    for i in range(Nb):
        def cfun(C, i=i):
            return C[i]
    
        def cjac(C, i=i):
            ret = np.zeros_like(C)
            ret[i]=1
            return ret 
        
        constr.append({
               
            "type": "ineq",
            "fun": cfun,
            "jac": cjac
            
            })
    return constr
    
def fit_N_radius(M, b, psquare, nspecies):
    """Find the best N-disperse radius
    
    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    nspecies: int
        Number of species to fit.
    
    Returns
    -------
    spectrum: 1d array
        The best radius fit spectrum
    """
    NRs = len(b)
    indices = np.asarray([i for i in combinations(range(NRs), nspecies)])
    res = np.empty(len(indices))
    C = np.empty((len(indices), nspecies))
    C0 = np.ones(nspecies)/nspecies
    best = psquare
    for i, idx in enumerate(indices):
        bi = b[idx]
        Mi = M[idx][:, idx]
        min_res = minimize(fun, C0, args=(Mi, bi, psquare),
                   jac=jac, hess=hess, 
                   constraints=getconstr(nspecies))
        if min_res.fun < best:
            best = min_res.fun
            print('New best: ',best)
        res[i] = min_res.fun
        C[i] = min_res.x
      
    bestidx = np.argmin(res)
    idx = indices[bestidx]
    spectrum = np.zeros(NRs)
    spectrum[idx] = C[bestidx]
    
    return spectrum
        
        
def fit_polydisperse_radius(M, b, psquare):
    """Find the best N-disperse radius
    
    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    
    Returns
    -------
    spectrum: 1d array
        The best fit spectrum
    """     
    
    Nb = len(b)
    C0 = np.zeros(Nb)

    def fun2(C, M, b, psquare):
        return fun(np.abs(C), M, b, psquare)
    
    def jac2(C, M, b, psquare):
        return jac(np.abs(C), M, b, psquare)*np.sign(C)
    
    res = basinhopping(fun2, C0, 100, disp=True,
                       minimizer_kwargs={'args': (M, b, psquare),
                                         'jac': jac2,
                                         })
    spectrum = np.abs(res.x)
    return spectrum

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
    prof = np.array(prof)
    prof[np.isnan(prof)] = 0
    Yi = prof[::-1]
    corr = np.correlate(prof,Yi, mode='full')
    X = np.arange(len(corr))
    args = np.argsort(corr)
    x = X[args[-7:]]
    y = corr[args[-7:]]
    coeffs = np.polyfit(x,np.log(y),2)
    center = -coeffs[1]/(2*coeffs[0])
    center = (center-(len(corr)-1)/2)/2+(len(prof)-1)/2
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
    argvalid = np.argwhere(np.isfinite(prof))
    lims = np.squeeze([argvalid[0],argvalid[-1]])
    left = int(lims[0]+frac*np.diff(lims))
    right = int(lims[1]-frac*np.diff(lims))
    leftZero = np.nanmean(prof[lims[0]:left])
    rightZero = np.nanmean(prof[right:lims[1]])
        
    #Send profile to 0
    baseline = np.linspace(leftZero,rightZero,len(prof))
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
    leftZero = np.nanmean(prof[:int(frac*len(prof))])
    rightZero = np.nanmean(prof[-int(frac*len(prof)):])
        
    #Send profile to 0
    ret = np.zeros(prof.shape)+np.mean([leftZero,rightZero])
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
    argvalid = np.argwhere(np.isfinite(np.nanmean(image,1)))
    lims = np.squeeze([argvalid[0],argvalid[-1]])
    #should we flatten this?
    top = np.nanmean(image[lims[0]:np.mean(lims,dtype=int)] ,0)
    bottom = np.nanmean(image[np.mean(lims,dtype=int):lims[1]],0)
    #Remouve nans
    top[np.isnan(top)] = 0
    bottom[np.isnan(bottom)] = 0
    #correlate
    C = np.correlate(top,bottom, mode='full')
    
    pos = np.arange(len(C))-(len(C)-1)/2
    disty = ((lims[1]-lims[0])/2)
    Angles = np.arctan(pos/disty)
    
    valid = np.abs(Angles)<maxAngle
    x = pos[valid]
    c = C[valid]  

    x = x[c.argmax()-5:c.argmax()+6]
    y = np.log(gfilter(c,2)[c.argmax()-5:c.argmax()+6])  
    
    if np.any(np.isnan(y)):
        raise RuntimeError('The signal is too noisy!')
        
    coeff = np.polyfit(x,y,2)
    x = -coeff[1]/(2*coeff[0])
    angle = np.arctan(x/disty)     
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arctan(pos/disty),C)
    plt.plot([maxAngle,maxAngle],[np.min(C),np.max(C)])
    plt.plot([-maxAngle,-maxAngle],[np.min(C),np.max(C)])
    plt.plot([angle,angle],[np.min(C),np.max(C)])
    #"""
    """
    import matplotlib.pyplot as plt
    x = np.arange(len(top))
    plt.figure()
    plt.plot(x,top)
    plt.plot(x+(C.argmax()-(len(C)-1)/2),bottom)
    plt.title('image angle')
    #"""
    return angle

def init_process(profile, mode, ignore):
    """
    Process the initial profile
    
    Parameters
    ----------
    profile:  1d array
        Profile to analyse 
    mode: string
        'none':
            Nothing
        'gaussian':
            Return a gaussian fit
        'tails':
            Remove the tails
        'gfilter':
            Apply a gaussian filter of 2 px std
    ignore: int or None
        The number of pixels to ignore on the edges
    Returns
    -------
    profile: 1d array
        the processed profile
    
    """
    profile = np.array(profile)
    if ignore is not None and ignore !=0:
        profile[:ignore]=0
        profile[-ignore:]=0

    if mode == 'none':
        return profile
    elif mode == 'gfilter':
        return gfilter(profile,2)
    elif mode == 'gaussian' or mode == 'tails':
        Y = profile
        X = np.arange(len(Y))
        valid = Y>.5*Y.max()
        gauss = np.exp(np.poly1d(np.polyfit(X[valid],np.log(Y[valid]),2))(X))
        if mode=='gaussian':
            return gauss
        remove = gauss<.01*gauss.max()
        profile[remove] = 0
        return profile

def get_fax(profiles):
    """
    returns a faxed verion of the profiles for easier plotting
    
    Parameters
    ----------
    profiles:  2d array
        List of profiles
        
    Returns
    -------
    profiles: 1d array
        The faxed profiles
    
    """
    return np.ravel(np.concatenate(
            (profiles,np.zeros((np.shape(profiles)[0],1))*np.nan),axis=1))
    
def get_edge(profile):
    """Get the largest edge in the profile
    
    Parameters
    ----------
    profile:  1d array
        profile to analyse
        
    Returns
    -------
    edgePos: float
        The edge position
    """
    e = np.abs(np.diff(gfilter(profile,2)))
    valid = slice(np.argmax(e)-3,np.argmax(e)+4)
    X = np.arange(len(e))+.5
    X = X[valid]
    Y = np.log(e[valid])
    coeff = np.polyfit(X,Y,2)
    edgePos = -coeff[1]/(2*coeff[0])
    return edgePos

def get_profiles(scans, Npix, orientation=None, *, 
                 offset_edge_idx =None, offset=0):
    """Extract profiles from scans
    
    Parameters
    ----------
    scans:  2d array
        sacns to analyse
    Npix:   integer
        number of pixels in a profile
    orientation: 1d array
        Orientation of each scan (Positive or negative)
    offset_edge_idx: integer
        Index of a profile containing an edge and a maximum to detect offset
    offset: integer
        Manual offset
        
    Returns
    -------
    profiles: 1d array
        The profiles
    """
    
    #Init return
    profiles = np.empty((scans.shape[0],Npix))
    scans = np.array(scans)
    if offset_edge_idx is not None and offset_edge_idx<0:
        offset_edge_idx = len(scans)+offset_edge_idx
    
    #Straighten scans
    if orientation is not None:
        for s,o in zip(scans,orientation):
            if o<0:
                s[:] = s[::-1]
    
    # get the offset if needed
    if offset_edge_idx is not None:
        offset_scan = scans[offset_edge_idx]
        cent = center(offset_scan)
        edge = get_edge(offset_scan)
        offset = np.abs(cent-edge)-Npix/2
        edgeside = 1
        if edge>cent:
            edgeside = -1
    
    #For each scan
    for i,s in enumerate(scans):
        #Get the mid point
        if offset_edge_idx is None:
            mid = center(s)-offset
        else:
            if i<offset_edge_idx:
                mid = center(s)-edgeside*offset
            else:
                mid = get_edge(s)+edgeside*Npix/2
        #First position
        amin = int(mid-Npix/2)
        #If pixels missings:
        if amin<0 or amin>len(s)-Npix:
            warnings.warn("Missing pixels, scan not large enough", 
                          RuntimeWarning)
            while amin>len(s)-Npix:
                s = np.append(s,s[-1])
            while amin<0:
                amin+=1
                s = np.append(s[0],s)
        #Get profile
        profiles[i] = s[amin:amin+Npix]
        
    return profiles
