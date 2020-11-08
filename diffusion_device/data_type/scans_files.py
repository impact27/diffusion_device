#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:13:33 2018

@author: quentinpeter
"""
import csv
import numpy as np
from ..profile import rebin_profiles


def load_file(fn, delimiter, skiprows, index, transpose=True, rebin=None):

    if not isinstance(fn, str):
        return np.asarray([load_file(
            name, delimiter, skiprows, index, transpose) for name in fn])

    index = tuple((slice(*val) for val in index))

    value = np.loadtxt(fn, delimiter=delimiter, skiprows=skiprows)[index]
    if transpose:
        value = value.T

    profiles = np.squeeze(value)
    if rebin:
        profiles = rebin_profiles(profiles.T, rebin).T
    return profiles


def save_file(fn, scan):
    scan = np.asarray(scan)
    if len(scan.shape) == 1:
        scan = scan[np.newaxis]
    # index = np.arange(np.shape(scan)[0])
    with open(fn, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(
            ['X'] + [f'Y{i}' for i in range(np.shape(scan)[0])])
        for i, line in enumerate(scan.T):
            csv_writer.writerow([f'{i}'] + [f'{n}' for n in line])
