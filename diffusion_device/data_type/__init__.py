# -*- coding: utf-8 -*-
"""
Folder of modules defining:

data = load_data(metadata, infos)
data = process_data(data, metadata, settings, infos)
profiles = get_profiles(data, metadata, settings, infos)
radius, fits = size_profiles(profiles, metadata, settings, infos)
plot_and_save(radius, profiles, fits, outpath, settings, infos)
savedata(data, outpath)

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
from .. import profile as dp
from .. import display_data


class DataType():
    def __init__(self, metadata, settings, outpath):
        """Init"""
        super().__init__()
        self.infos = {}
        self.metadata = metadata
        self.settings = settings
        self.outpath = outpath

    def full_fit(self):
        """Perform a fit with the imformations found in the settings file"""
        raw_data = self.load_data()
        data = self.process_data(raw_data)

        if self.outpath is not None:
            self.savedata(data)

        profiles = self.get_profiles(data)
        profiles = self.process_profiles(profiles)
        radius, fits = self.size_profiles(profiles)

        if self.outpath is not None:
            self.plot_and_save(radius, profiles, fits)

        return radius, profiles, fits, data, self.infos

    def load_data(self):
        """load data from metadata

        Returns
        -------
        data: array
            the image
        """
        assert(False)

    def process_data(self, data):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process

        Returns
        -------
        data: array
            The processed data
        """
        return data

    def savedata(self, data):
        """Save the data"""
        pass

    def get_profiles(self, data):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process

        Returns
        -------
        profiles: array
            The profiles
        """
        assert(False)

    def process_profiles(self, profiles):
        """process_profiles"""
        return profiles

    def size_profiles(self, profiles):
        """size_profiles"""
        return dp.size_profiles(
            profiles, self.metadata, self.settings, infos=self.infos)

    def plot_and_save(self, radius, profiles, fits):
        """plot_and_save"""
        display_data.plot_and_save(
            radius, profiles, fits, self.infos, self.outpath)
