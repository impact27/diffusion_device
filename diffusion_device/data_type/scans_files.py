#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:13:33 2018

@author: quentinpeter
"""
import numpy as np
from os.path import splitext

def load_file(fn):
    
    if not isinstance(fn, str):
        return np.asarray([load_file(name) for name in fn])
    
    ext = splitext(fn)[1]
    if ext == '.csv':
        return np.loadtxt(fn, delimiter=',')[:, 1]
    elif ext == '.dat':
        return np.loadtxt(fn, skiprows=22)
    else:
        raise RuntimeError(f'Unknown extension {ext}')