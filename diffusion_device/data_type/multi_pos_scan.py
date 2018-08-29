# -*- coding: utf-8 -*-
"""
Analyse confocal scans

Created on Tue Sep 12 13:33:18 2017

@author: quentinpeter

Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.ndimage.measurements import label
from scipy import interpolate
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.signal import savgol_filter

from .. import profile as dp
from . import DataType
from .scans_files import load_file

class MultiPosScan(DataType):

    def __init__(self, metadata, settings, outpath):
        super().__init__(metadata, settings, outpath)

    def load_data(self, filename=None):
        """load data from metadata

        Parameters
        ----------
        metadata: dict
            The metadata information
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        data: array
            the image
        """
        if filename is None:
            filename = self.metadata["KEY_MD_FN"]
        data = load_file(filename)
        
        # Apply scan slice
        scan_slice = self.settings["KEY_SET_SCAN_SLICE"]
        if scan_slice is not None:
            data = data[scan_slice[0]:scan_slice[1]]
        return data

    
    def process_data(self, raw_data):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process
        metadata: dict
            The metadata information
        settings: dict
            The settings
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        data: array
            The processed data
        """
        background_fn = self.metadata["KEY_MD_BGFN"]

        # Find centers for processing
        centers, pixel_size = self.get_scan_centers(raw_data)
        self.infos["Pixel size"] = pixel_size
        self.infos["Centers"] = centers
        self.infos["flow direction"] = self.metadata["KEY_MD_FLOWDIR"]

        # Get background
        if background_fn is None:
            data = self.flatten_scan(raw_data)
        else:
            bg = self.load_data(background_fn)
            data = self.remove_scan_background(raw_data, bg)
            
        self.noise_var = self.get_noise_var(raw_data)
        return data

    def flatten_scan(self, data):
        """flatten_scan"""
        out_mask = self.get_out_mask(data)
        data -= np.nanmean(data[out_mask])
        return data
    
    def get_channel_mask(self, data):
        """get_channel_mask"""
        
        channel_width = self.metadata["KEY_MD_WY"]
        pixel_size = self.infos["Pixel size"]
        centers = self.infos["Centers"]
        
        X = np.arange(len(data))
        channel_mask = np.any(np.abs(
                X[:, np.newaxis] - centers[np.newaxis]
            ) * pixel_size < channel_width/2, axis=1)
        return channel_mask
    
    def get_out_mask(self, data):
        """get_out_mask"""
        
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        pixel_size = self.infos["Pixel size"]
        centers = self.infos["Centers"]
        
        out_mask = np.logical_not(self.get_channel_mask(data))
        
        #Remove what is too far
        X = np.arange(len(data))
        far_mask = np.min(np.abs(
            X[:, np.newaxis] - centers[np.newaxis]
        ), axis=1) * pixel_size > channel_width/2 + wall_width
        
        out_mask[far_mask] = False
        
        return out_mask
    
    def get_noise_var(self, raw_data):
        """get_noise_var"""
        noise_var = raw_data.copy()
        if 'offset' in self.infos:
            offset = self.infos['offset'] 
            if offset < 0:
                noise_var = noise_var[:offset]
                
            else:
                noise_var = noise_var[offset:]
            
        return noise_var
        
    def remove_scan_background(self, data, bg):
        """remove_scan_background"""
        
        centers = self.infos["Centers"]
        # Determine what is in the channels and far from the channels
        out_mask = self.get_out_mask(data)

        # Get dummy profiles and correlate
        p0 = data - np.mean(data[out_mask])
        p1 = bg - np.mean(bg)
        p0[np.logical_not(out_mask)] = 0
        corr = np.correlate(np.tile(p0, 2), p1, mode='valid')[:-1]

        # Get offset and apply
        offset = np.argmax(corr)
        if offset > len(p0)/2:
            offset -= len(p0)
        self.infos['offset'] = offset 
        if offset < 0:
            bg = bg[-offset:]
            data = data[:offset]
            
        else:
            bg = bg[:-offset]
            data = data[offset:]
            centers -= offset
            self.infos["Centers"] = centers

        # Get updated mask
        out_mask = self.get_out_mask(data)

        # Scale background
        newbg = bg * (np.sum(data[out_mask] * bg[out_mask]) 
                    / np.sum(bg[out_mask] * bg[out_mask]))

        # Subtract
        data = data - newbg
        return data
        
    def get_profiles(self, lin_profiles):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process
        metadata: dict
            The metadata information
        settings: dict
            The settings
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        profiles: array
            The profiles
        """
        profiles = self.extract_profiles(lin_profiles)

        profiles, self.infos["Pixel size"] = dp.process_profiles(
            profiles, self.metadata, self.settings,
            self.outpath, self.infos["Pixel size"])

        profiles = self.align_profiles(profiles, lin_profiles)

        profiles, self.infos["Pixel size"] = dp.process_profiles(
            profiles, self.metadata, self.settings,
            self.outpath, self.infos["Pixel size"])

        return profiles

    def max_to_center(self, maxs):
        """Get centers from a max distribution"""
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
         
        number_profiles = len(maxs)
        # Get distances
        dist = np.abs(np.diff(maxs))
        dist_even = np.mean(dist[::2])
        dist_odd = np.mean(dist[1::2])
        meandist = 1/2*(dist_even + dist_odd)

        # Correct for any off balance
        centers = np.asarray(maxs, float)
        centers[::2] += (dist_even - meandist) / 2
        centers[1::2] += (dist_odd - meandist) / 2

        # Get evenly spaced centers
        start = np.mean(centers - np.arange(number_profiles) * meandist)
        centers = start + np.arange(number_profiles) * meandist

        pixel_size = np.abs((wall_width + channel_width) / meandist)
        return centers, pixel_size

    def get_scan_centers(self, profiles):
        """Get centers from a single scan"""
        
        number_profiles = self.metadata["KEY_MD_NCHANNELS"]
        
        profiles = profiles - np.nanmin(profiles)
        profiles[np.isnan(profiles)] = 0

        # Filter heavely and get position of the centers as a first approx.
        filter_width = len(profiles)/((number_profiles*2+1)*3)
        Hfiltered = gfilter(profiles, filter_width)
        maxs = np.where(maximum_filter1d(
            Hfiltered, int(filter_width)) == Hfiltered)[0]

        # Filter lightly and get 2nd derivative
        fprof = gfilter(profiles, 3)
        # If max negatuve, not a max
        maxs = maxs[profiles[maxs] > 0]
        # If filter reduces int by 50%, probably a wall
        maxs = maxs[(profiles[maxs] - fprof[maxs]) / profiles[maxs] < .5]
        # Remove sides
        maxs = maxs[np.logical_and(
            maxs > 3/2*filter_width, maxs < len(fprof) - 3/2*filter_width)]
        maxs = maxs[np.argsort(fprof[maxs])[- number_profiles:]][::-1]

        # Sort and check number
        maxs = sorted(maxs)
        if len(maxs) != number_profiles:
            raise RuntimeError("Can't find the center of the channels!")

        centers, pixel_size = self.max_to_center(maxs)

    #    from matplotlib.pyplot import figure, show, plot, imshow, title
    #    plot(profiles); plot(centers, np.zeros(len(centers)), 'x'); show()

        return centers, pixel_size

    def should_switch(self, flow_direction):
        if flow_direction == 'u' or flow_direction == 'up':
            switch = True
        elif flow_direction == 'd' or flow_direction == 'down':
            switch = False
        else:
            raise RuntimeError("unknown orientation: {}".format(
                flow_direction))
        return switch

    def interpolate_profiles(self, lin_profiles, centers, flowdir, prof_npix,
                             prof_width, old_pixel_size):
        new_pixel_size = prof_width/prof_npix
        nchannels = len(centers)
        profiles = np.empty((nchannels, prof_npix), dtype=float)

        # Extract profiles
        for i, (cent, fd) in enumerate(zip(centers, flowdir)):

            X = np.arange(len(lin_profiles)) - cent
            finterp = interpolate.interp1d(X * old_pixel_size, lin_profiles)

            Xc = np.arange(prof_npix) - (prof_npix - 1) / 2
            p = finterp(Xc * new_pixel_size)

            if self.should_switch(fd):
                p = p[::-1]

            profiles[i] = p
        return profiles, new_pixel_size

    def extract_profiles(self, lin_profiles):
        """Extract profiles from a single scan"""
        
        channel_width = self.metadata["KEY_MD_WY"]
        centers = self.infos["Centers"]
        flowdir = self.infos["flow direction"]
        pixel_size = self.infos["Pixel size"]

        prof_npix = int(np.round(channel_width / pixel_size))

        if (np.min(centers) - prof_npix / 2 < 0 or
                np.max(centers) + prof_npix / 2 > len(lin_profiles)):
            raise RuntimeError('Channel not fully contained in the image')

        profiles, pixel_size = self.interpolate_profiles(
            lin_profiles, centers, flowdir, prof_npix,
            prof_npix * pixel_size, pixel_size)

        # If image upside down, turn
        if profiles[-1].max() > profiles[0].max():
            profiles = profiles[::-1]
            centers = centers[::-1]
            flowdir = flowdir[::-1]
            
        self.infos["Centers"] = centers
        self.infos["flow direction"] = flowdir
        self.infos["Pixel size"] = pixel_size
        self.infos["Profiles noise std"] = self.get_noise(
                lin_profiles, self.noise_var)
        return profiles
    
    def get_noise(self, lin_profiles, lin_var, prof_npix=None):
        """get_noise"""
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        centers = self.infos["Centers"]
        
        if prof_npix is None:
            pixel_size = (channel_width + wall_width) / np.abs(np.mean(np.diff(centers)))
            prof_npix = int(np.round(channel_width / pixel_size))
        
        outmask = np.all(
            np.abs(np.arange(len(lin_profiles))[:, np.newaxis]
                   - centers[np.newaxis]) > .55 * prof_npix, axis=1)
        outmask = np.logical_and(outmask, np.isfinite(lin_profiles))
        outmask = np.logical_and(outmask, np.isfinite(lin_var))

        lbl, n = label(outmask)
        wall_var = np.zeros(n)
        lin_var_list = np.zeros(n)

        for i in np.arange(n):
            mask = lbl == i + 1
            background = lin_profiles[mask]
            window = 31
            if len(background) < 3:
                wall_var[i] = np.sum(np.square(background
                                           - np.median(background)))
            else:
                if len(background) < window:
                    window = 2*(len(background)//2) - 1
                wall_var[i] = np.sum(np.square(
                    background
                    - savgol_filter(background, window, window//6)))
            lin_var_list[i] = np.sum(lin_var[mask])

        noise = lin_var / np.sum(lin_var_list) * np.sum(wall_var)
        noise, __ = self.interpolate_profiles(
            noise, centers, self.infos["flow direction"],
            prof_npix, channel_width, self.infos["Pixel size"])
        
        rebin = self.settings["KEY_STG_REBIN"]
        noise = dp.rebin_profiles(noise, rebin)/rebin
        
        min_var = np.sum(wall_var) / np.sum(outmask)
        noise[ noise < min_var] = min_var
        # if True:
        #     noise_std = noise_std * 0 + np.sqrt(np.sum(wall_var) / np.sum(outmask))
        return np.sqrt(noise)

    def align_profiles(self, profiles, lin_profiles):
        ignore = self.settings["KEY_STG_IGNORE"]
        rebin = self.settings["KEY_STG_REBIN"]
        pixel_size = self.infos["Pixel size"]
        centers = self.infos["Centers"]
        flowdir = self.infos["flow direction"]
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]

        prof_npix = np.shape(profiles)[1] * rebin

        __, fits = dp.size_profiles(profiles,
                                    self.metadata, self.settings, self.infos)

        profile_slice = dp.ignore_slice(ignore, pixel_size)

        new_centers = np.array(centers, dtype=float)
        for i, (cent, fd) in enumerate(zip(centers, flowdir)):

            # Get diff between profile and fit
            diff = dp.center(np.correlate(
                profiles[i, profile_slice],
                fits[i, profile_slice],
                mode='full')) - len(profiles[i, profile_slice]) + 1
            if not np.isnan(diff):
                diff *= rebin
                switch = self.should_switch(fd)
    
                if switch:
                    diff *= -1
    
                new_centers[i] = cent + diff

        meandist = np.mean(np.diff((new_centers[1:] + new_centers[:-1])/2))
        pixel_size = np.abs((channel_width + wall_width) / meandist)

        res = (new_centers - np.min(new_centers) + meandist/2) % meandist
        offset = (np.mean(res[::2]) - np.mean(res[1::2])) / 2

        new_centers -= offset * (-1)**np.arange(len(profiles))

        new_profiles, pixel_size = self.interpolate_profiles(
            lin_profiles, new_centers, flowdir,
            prof_npix, channel_width, pixel_size)

        self.infos["Centers"] = new_centers
        self.infos["Pixel size"] = pixel_size
        self.infos["Profiles noise std"] = self.get_noise(
                lin_profiles, self.noise_var, prof_npix=prof_npix)
        return new_profiles
