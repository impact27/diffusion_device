# -*- coding: utf-8 -*-
"""
Module for movies of 4 pos channels

Created on Wed Sep 13 10:15:41 2017

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

from registrator.image import is_overexposed
import numpy as np
import tifffile
import sys

from .multi_pos_image import MultiPosImage
from .. import display_data
from . import images_files
from .. import profile as dp


class StackMultiPosImage(MultiPosImage):

    def __init__(self, metadata, settings, outpath):
        super().__init__(metadata, settings, outpath)

    def load_data(self):
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
        filename = self.metadata["KEY_MD_FN"]
        data = self.load_images(filename)
        self.infos["Overexposed"] = [is_overexposed(d) for d in data]
        return data

    def process_data(self, data):
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
        Nchannel = self.metadata['KEY_MD_NCHANNELS']
        framesslices = slice(*self.settings["KEY_STG_STACK_FRAMESSLICES"])
        
        data = np.asarray(data[framesslices], dtype="float32")
        
        rebin = self.settings["KEY_STG_STACK_REBIN"]
        if rebin > 1:
            new_data = np.zeros(
                (np.shape(data)[0]//rebin, *np.shape(data)[1:]))
            for i in range(len(new_data)):
                new_data[i] = np.mean(data[i * rebin:(i + 1) * rebin], 0)
            data = new_data

            new_overexposed = np.zeros(len(self.infos["Overexposed"])//rebin, bool)
            for i in range(len(new_overexposed)):
                new_overexposed[i] = np.any(
                    self.infos["Overexposed"][i * rebin:(i + 1) * rebin])

            self.infos["Overexposed"] = new_overexposed
            for i, val in enumerate(self.settings["KEY_STG_STACK_FRAMESSLICES"]):
                if val is not None:
                    self.settings["KEY_STG_STACK_FRAMESSLICES"][i] = val // rebin
            framesslices = slice(*self.settings["KEY_STG_STACK_FRAMESSLICES"])

        centers = np.zeros((len(data), Nchannel))
        pixel_size = np.zeros((len(data)))
        dataout = []
        skip = []
        noise_var = []

        if self.settings["KEY_STG_STAT_STACK"]:
            # Check KEY_MD_EXP are all the same
            if isinstance(self.metadata["KEY_MD_EXP"], list):
                if np.all(np.equal(self.metadata["KEY_MD_EXP"],
                                   self.metadata["KEY_MD_EXP"][0])):
                    raise RuntimeError(
                        "Can not have different exposure times"
                        " when using stationary option.")
                else:
                    self.metadata["KEY_MD_EXP"] = self.metadata["KEY_MD_EXP"][0]

            return super().process_data(data)

        infos_stack = self.infos
        metadata_stack = self.metadata.copy()
        for i in range(len(data)):
            try:
                # Get KEY_MD_EXP correct in the metadata
                if isinstance(metadata_stack["KEY_MD_EXP"], list):
                    self.metadata["KEY_MD_EXP"] = (
                        np.asarray(metadata_stack["KEY_MD_EXP"])[framesslices][i])
                d = super().process_data(data[i])

                pixel_size[i] = self.infos["Pixel size"]
                centers[i] = self.infos["Centers"]
                noise_var.append(self.infos["noise_var"])
                dataout.append(d)
            except BaseException:
                if self.settings["KEY_STG_IGNORE_ERROR"]:
                    print(sys.exc_info())
                    pixel_size[i] = np.nan
                    centers[i, :] = np.nan
                    skip.append(i)
                    noise_var.append(None)
                else:
                    raise

        infos_stack["flow direction"] = self.infos["flow direction"]
        self.infos = infos_stack
        # Fix metadata
        metadata_stack["KEY_MD_FLOWDIR"] = self.metadata["KEY_MD_FLOWDIR"]
        self.metadata = metadata_stack
        dataout = np.asarray(dataout)
        if skip != []:
            for idx in skip:
                dataout = np.insert(dataout, idx,
                                    np.ones(np.shape(dataout)[1:]) * np.nan, 0)

        self.infos["Pixel size"] = pixel_size
        self.infos["Centers"] = centers
        self.infos["noise_var"] = noise_var
        return dataout

    def get_profiles(self, data):
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
        pixel_size = self.infos["Pixel size"]
        centers = self.infos["Centers"]
        flowdir = self.infos["flow direction"]
        profiles = []
        noises = []
        infos_tmp = self.infos

        for i, im in enumerate(data):
            try:
                if self.settings["KEY_STG_STAT_STACK"]:
                    pxs, cnt = pixel_size, centers
                else:
                    pxs, cnt = pixel_size[i], centers[i]

                infos_i = {
                    "Pixel size": pxs,
                    "Centers": cnt,
                    "flow direction": flowdir,
                    'noise_var': infos_tmp['noise_var'][i]}

                self.infos = infos_i
                prof = super().get_profiles(im)
                noise = infos_i["Profiles noise std"]
            except BaseException:
                if self.settings["KEY_STG_IGNORE_ERROR"]:
                    print(sys.exc_info())
                    prof = None
                    noise = None
                else:
                    raise
            profiles.append(prof)
            noises.append(noise)

        self.infos = infos_tmp
        self.infos["Profiles noise std"] = noises
        return profiles

    def get_infos_i(self, i):
        """get_infos_i"""
        if self.settings["KEY_STG_STAT_STACK"]:
            infos_i = {
                "Pixel size": self.infos["Pixel size"]}
        else:
            infos_i = {
                "Pixel size": self.infos["Pixel size"][i]}
        infos_i["Profiles noise std"] = self.infos["Profiles noise std"][i]
        return infos_i

    def set_infos_i(self, infos_i, i):
        """set_infos_i"""
        if self.settings["KEY_STG_STAT_STACK"]:
            self.infos["Pixel size"] = infos_i["Pixel size"]
        else:
            self.infos["Pixel size"][i] = infos_i["Pixel size"]
        self.infos["Profiles noise std"][i] = infos_i["Profiles noise std"]

    def size_profiles(self, profiles):
        """Size the profiles

         Parameters
        ----------
        profiles: 2d array
            List of profiles to fit
        pixel_size:float
            The pixel size in [m]
        metadata: dict
            The metadata
        settings: dict
            The settings

        Returns
        -------
        radius:
            if nspecies==1:
                radii: float
                    The best radius fit
            else:
                Rs, spectrum, the radii and corresponding spectrum
        fits: 2d array
            The fits
        """
        radius = []
        fits = []
        errors = []
        r_errors = []
        r_ranges = []
        shape_r = None
        signal_over_noise = np.zeros(len(profiles)) * np.nan
        for i, profs in enumerate(profiles):
            r = None
            fit = None
            error = None
            r_error = None
            r_range = None
            if profs is not None:
                try:
                    infos_i = self.get_infos_i(i)
                    if len(np.shape(self.metadata["KEY_MD_Q"])) > 0:
                        metadata_i = self.metadata.copy()
                        metadata_i["KEY_MD_Q"] = metadata_i["KEY_MD_Q"][i]
                    else:
                        metadata_i = self.metadata
                    r, fit = dp.size_profiles(
                        profs, metadata_i, self.settings, infos_i)
                    shape_r = np.shape(r)
                    error = infos_i["Reduced least square"]
                    r_error = infos_i["Radius error std"]
                    signal_over_noise[i] = infos_i["Signal over noise"]
                    r_range = infos_i["Radius range"]
                except BaseException:
                    if self.settings["KEY_STG_IGNORE_ERROR"]:
                        print(sys.exc_info()[1])
                    else:
                        raise
            fits.append(fit)
            radius.append(r)
            errors.append(error)
            r_errors.append(r_error)
            r_ranges.append(r_range)
            
        if shape_r is None:
            raise RuntimeError("Can't find a single good frame")

        self.infos["Reduced least square"] = errors
        self.infos["Radius error std"] = r_errors
        self.infos["Radius range"] = r_ranges
        self.infos["Signal over noise"] = signal_over_noise
        return radius, fits

    def savedata(self, data):
        """Save the data"""
        tifffile.imsave(self.outpath + '_ims.tif', np.asarray(data, 'float32'))

    def plot_and_save(self, radius, profiles, fits):
        """Plot the sizing data"""
        framesslices = slice(*self.settings["KEY_STG_STACK_FRAMESSLICES"])

        self.infos["Overexposed"] = self.infos["Overexposed"][framesslices]

        display_data.plot_and_save_stack(
            radius, profiles, fits, self.infos, self.settings, self.outpath)

    def process_profiles(self, profiles):
        ret = []
        for i, prof in enumerate(profiles):
            if prof is None:
                ret.append(None)
            else:
                infos = self.get_infos_i(i)
                prof = dp.process_profiles(
                    prof, self.metadata, self.settings, self.outpath, infos)
                ret.append(prof)
                self.set_infos_i(infos, i)

        return ret
