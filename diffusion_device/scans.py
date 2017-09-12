# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:33:18 2017

@author: quentinpeter
"""
import numpy as np
import warnings
from scipy.ndimage.filters import gaussian_filter1d as gfilter

from . import profile as dp
from . import keys

def process_scans(scans, metadata):
    flow_dir = metadata[keys.KEY_MD_FLOWDIR]
    
    # put scans in the correct order
    if flow_dir is not None:
        for s, o in zip(scans, flow_dir):
            if o == 'u':
                s[:] = s[::-1]
            elif o != 'd':
                raise RuntimeError(
                        'Flow direction must be up or down for scans.')
                
    return scans
                
def get_profiles(scans, Npix, *,
                 offset_edge_idx=None):
    """Extract profiles from scans

    Parameters
    ----------
    scans:  2d array
        sacns to analyse
    Npix:   integer
        number of pixels in a profile
    offset_edge_idx: integer
        Index of a profile containing an edge and a maximum to detect offset
    offset: integer
        Manual offset

    Returns
    -------
    profiles: 1d array
        The profiles
    """
    offset = 0
    # Init return
    profiles = np.empty((scans.shape[0], Npix))
    scans = np.array(scans)
    if offset_edge_idx is not None and offset_edge_idx < 0:
        offset_edge_idx = len(scans) + offset_edge_idx

    # get the offset if needed
    if offset_edge_idx is not None:
        offset_scan = scans[offset_edge_idx]
        cent = dp.center(offset_scan)
        edge = get_edge(offset_scan)
        offset = np.abs(cent - edge) - Npix / 2
        edgeside = 1
        if edge > cent:
            edgeside = -1

    # For each scan
    for i, s in enumerate(scans):
        # Get the mid point
        if offset_edge_idx is None:
            mid = dp.center(s) - offset
        else:
            if i < offset_edge_idx:
                mid = dp.center(s) - edgeside * offset
            else:
                mid = get_edge(s) + edgeside * Npix / 2
        # First position
        amin = int(mid - Npix / 2)
        # If pixels missings:
        if amin < 0 or amin > len(s) - Npix:
            warnings.warn("Missing pixels, scan not large enough",
                          RuntimeWarning)
            while amin > len(s) - Npix:
                s = np.append(s, s[-1])
            while amin < 0:
                amin += 1
                s = np.append(s[0], s)
        # Get profile
        profiles[i] = s[amin:amin + Npix]

    return profiles

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
    diff = np.diff(profile)
    
    diffnorm = diff / (profile[:-1]+profile[1:])
    
    left_edge = getmaxaround(diff, np.argmax(diffnorm)) + .5
    right_edge = getmaxaround(diff, np.argmin(diffnorm)) + .5
    
    return left_edge#, right_edge
    

def getmaxaround(profile, approxmax, window_r = 3):
    valid = slice(approxmax - window_r, approxmax + window_r +1)
    X = np.arange(len(profile)) 
    X = X[valid]
    Y = np.log(profile[valid])
    coeff = np.polyfit(X, Y, 2)
    edgePos = -coeff[1] / (2 * coeff[0])
    return edgePos
    
#def get_profiles(scans, Npix, *,
#                 offset_edge_idx=None):
#    """Extract profiles from scans
#
#    Parameters
#    ----------
#    scans:  2d array
#        sacns to analyse
#    Npix:   integer
#        number of pixels in a profile
#    offset_edge_idx: integer
#        Index of a profile containing an edge and a maximum to detect offset
#    offset: integer
#        Manual offset
#
#    Returns
#    -------
#    profiles: 1d array
#        The profiles
#    """
#    offset=0
#    # Init return
#    profiles = np.empty((scans.shape[0], Npix))
#    scans = np.array(scans)
#    if offset_edge_idx is not None and offset_edge_idx < 0:
#        offset_edge_idx = len(scans) + offset_edge_idx
#
#    # get the offset if needed
#    if offset_edge_idx is not None:
#        offset_scan = scans[offset_edge_idx]
#        cent = dp.center(offset_scan)
#        edge = get_edge(offset_scan)
#        offset = np.abs(cent - edge) - Npix / 2
#        edgeside = 1
#        if edge > cent:
#            edgeside = -1
#
#    # For each scan
#    for i, s in enumerate(scans):
#        # Get the mid point
#        if offset_edge_idx is None:
#            mid = dp.center(s) - offset
#        else:
#            if i < offset_edge_idx:
#                mid = dp.center(s) - edgeside * offset
#            else:
#                mid = get_edge(s) + edgeside * Npix / 2
#        # First position
#        amin = int(mid - Npix / 2)
#        # If pixels missings:
#        if amin < 0 or amin > len(s) - Npix:
#            warnings.warn("Missing pixels, scan not large enough",
#                          RuntimeWarning)
#            while amin > len(s) - Npix:
#                s = np.append(s, s[-1])
#            while amin < 0:
#                amin += 1
#                s = np.append(s[0], s)
#        # Get profile
#        profiles[i] = s[amin:amin + Npix]
#
#    return profiles