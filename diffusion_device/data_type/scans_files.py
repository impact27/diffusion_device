#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:13:33 2018

@author: quentinpeter
"""
from os.path import splitext
import csv
import numpy as np


def load_file(fn):

    if not isinstance(fn, str):
        return np.asarray([load_file(name) for name in fn])

    ext = splitext(fn)[1]
    if ext == '.csv':
        return np.squeeze(np.loadtxt(fn, delimiter=',', skiprows=1)[:, 1:].T)
    if ext == '.dat':
        return np.loadtxt(fn, skiprows=22)
    raise RuntimeError(f'Unknown extension {ext}')

def save_file(fn, scan):
    scan = np.asarray(scan)
    if len(scan.shape) == 1:
        scan = scan[np.newaxis]
    index = np.arange(np.shape(scan)[0])
    with open(fn, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['X'] + [f'Y{i}' for i in range(np.shape(scan)[0])])
        for i, line in enumerate(scan.T):
            csv_writer.writerow([f'{i}'] + [f'{n}' for n in line])
