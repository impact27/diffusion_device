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
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d, median_filter
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.signal import savgol_filter

from .. import profile as dp
from . import DataType
from .scans_files import load_file, save_file


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
        data = load_file(filename, **self.metadata["KEY_MD_SCAN_STRUCT"])

        # Apply scan slice
        scan_slice = self.settings["KEY_SET_SCAN_SLICE"]
        if scan_slice is not None:
            data = data[scan_slice[0]:scan_slice[1]]
        return data

    def savedata(self, infos):
        """Save the data"""
        save_file(self.outpath + '_scan.csv', infos['Data'])

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
        infos = {}
        infos["Pixel size"] = pixel_size
        infos["Centers"] = centers
        infos["flow direction"] = self.metadata["KEY_MD_FLOWDIR"]
        infos['Inlet location'] = self.metadata["KEY_MD_INLET_LOC"]

        # Get background
        if background_fn is None and not self.metadata["KEY_MD_BRIGHTWALL"]:
            data = self.flatten_scan(raw_data, infos)
        elif background_fn is not None:
            bg = self.load_data(background_fn)
            data = self.remove_scan_background(raw_data, bg, infos)
        else:
            data = raw_data

        infos["noise_var"] = self.get_noise_var(raw_data, infos)
        infos['Data'] = data
        return infos

    def flatten_scan(self, data, infos):
        """flatten_scan"""
        out_mask = self.get_out_mask(data, infos)
        data = data - np.nanmean(data[out_mask])
        return data

    def get_channel_mask(self, data, infos):
        """get_channel_mask"""

        channel_width = self.metadata["KEY_MD_WY"]
        pixel_size = infos["Pixel size"]
        centers = infos["Centers"]

        X = np.arange(len(data))
        channel_mask = np.any(np.abs(
            X[:, np.newaxis] - centers[np.newaxis]
        ) * pixel_size < channel_width / 2, axis=1)
        return channel_mask

    def get_out_mask(self, data, infos):
        """get_out_mask"""

        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        pixel_size = infos["Pixel size"]
        centers = infos["Centers"]

        out_mask = np.logical_not(self.get_channel_mask(data, infos))

        # Remove what is too far
        X = np.arange(len(data))
        far_mask = np.min(np.abs(
            X[:, np.newaxis] - centers[np.newaxis]
        ), axis=1) * pixel_size > channel_width / 2 + wall_width

        out_mask[far_mask] = False

        return out_mask

    def get_noise_var(self, raw_data, infos):
        """get_noise_var"""
        noise_var = raw_data.copy()
        if 'offset' in infos:
            offset = infos['offset']
            if offset < 0:
                noise_var = noise_var[:offset]

            else:
                noise_var = noise_var[offset:]

        return noise_var

    def remove_scan_background(self, data, bg, infos):
        """remove_scan_background"""

        centers = infos["Centers"]
        # Determine what is in the channels and far from the channels
        out_mask = self.get_out_mask(data, infos)

        # Get dummy profiles and correlate
        p0 = data - np.mean(data[out_mask])
        p1 = bg - np.mean(bg)
        p0[np.logical_not(out_mask)] = 0
        corr = np.correlate(np.tile(p0, 2), p1, mode='valid')[:-1]

        # Get offset and apply
        offset = np.argmax(corr)
        if offset > len(p0) / 2:
            offset -= len(p0)
        infos['offset'] = offset
        if offset < 0:
            bg = bg[-offset:]
            data = data[:offset]

        else:
            bg = bg[:-offset]
            data = data[offset:]
            centers -= offset
            infos["Centers"] = centers

        # Get updated mask
        out_mask = self.get_out_mask(data, infos)

        # Scale background
        newbg = bg * (np.sum(data[out_mask] * bg[out_mask])
                      / np.sum(bg[out_mask] * bg[out_mask]))

        # Subtract
        data = data - newbg
        return data

    def get_profiles(self, infos):
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
        infos["Profiles"] = self.extract_profiles(infos)

        infos = dp.process_profiles(
            infos, self.metadata, self.settings,
            self.outpath)

        infos = self.align_profiles(infos)

        infos = dp.process_profiles(
            infos, self.metadata, self.settings,
            self.outpath)

        return infos

    def max_to_center(self, maxs):
        """Get centers from a max distribution"""
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]

        number_profiles = len(maxs)
        # Get distances
        dist = np.abs(np.diff(maxs))
        dist_even = np.mean(dist[::2])
        dist_odd = np.mean(dist[1::2])
        meandist = 1 / 2 * (dist_even + dist_odd)

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
        brightwalls = self.metadata["KEY_MD_BRIGHTWALL"]
        wall_width_m = self.metadata["KEY_MD_WALLWIDTH"]
        channel_width_m = self.metadata["KEY_MD_WY"]
        pix_size_init = self.metadata["KEY_MD_PIXSIZE"]

        n_pixel_estimated = (
            ((wall_width_m + channel_width_m) * number_profiles - wall_width_m)
            / pix_size_init)

        def plot_pixel_size_helper():
            import matplotlib.pyplot as plt
            expected_distance = ((wall_width_m + channel_width_m)
                                 * (number_profiles - 1))
            plt.figure()
            plt.plot(profiles)
            plt.title('First peak to last peak distance = {:.2f}um'.format(
                expected_distance * 1e6))
            plt.xlabel('Position [px]')
            plt.show()


        if n_pixel_estimated < len(profiles) / 2:
            plot_pixel_size_helper()
            raise RuntimeError(
                'The channels are expected to cover less than half the image. '
                'Check pixel size or crop the image.')

        if n_pixel_estimated > len(profiles):
            plot_pixel_size_helper()
            raise RuntimeError(
                'The channels are expected to cover more than the image. '
                'Check pixel size and verify the image is not cropped.')

        if number_profiles < 3:
            raise RuntimeError('Need at least 3 profiles.')

        # Subrtact min, remove nans
        profiles = profiles - np.nanmin(profiles[1:-1])
        profiles[profiles < 0] = 0
        profiles[np.isnan(profiles)] = 0

        if brightwalls:
            centers, pix_size = self.find_center_brightwall(profiles)
        else:
            centers, pix_size = self.find_center_by_mask(profiles)
        return self.correlate_profiles(profiles, centers, pix_size)

    def find_center_by_mask(self, profiles):
        """Find the center and pixel sixe by estimating the mass."""
        number_profiles = self.metadata["KEY_MD_NCHANNELS"]
        wall_width_m = self.metadata["KEY_MD_WALLWIDTH"]
        channel_width_m = self.metadata["KEY_MD_WY"]
        pix_size_init = self.metadata["KEY_MD_PIXSIZE"]

        def find_channels(profiles, pix_size):
            """
            Try to find the channels by correlating a mask of the channels
            with the profiles
            """
            wall_width = wall_width_m / pix_size
            channel_width = channel_width_m / pix_size

            # Remove dust, bright peaks (10% median filter)
            small_filter_width = int(round(channel_width / 5))
            if small_filter_width > 3:
                profiles = median_filter(profiles, small_filter_width)

            # Create channel mask
            channel_mask = np.zeros(int(round(number_profiles * channel_width
                                      + (number_profiles) * wall_width)))
            offset_start = int(round(wall_width)) // 2
            for i in range(number_profiles):
                left = (offset_start
                        + i * int(round(wall_width + channel_width)))
                right = left + int(round(channel_width))
                channel_mask[left:right] = 1
            channel_mask = channel_mask > 0
            walls_mask = np.logical_not(channel_mask)

            channels_locator = gfilter(channel_mask * 1., channel_width / 4)
            channels_locator[walls_mask] = 0
            channels_locator /= np.sum(channels_locator)
            walls_locator = gfilter(walls_mask * 1., wall_width / 4)
            walls_locator[channel_mask] = 0
            walls_locator /= np.sum(walls_locator)

            cprofiles = np.zeros(len(profiles) + 2 * offset_start)
            cprofiles[offset_start:-offset_start] = profiles

            if len(channels_locator) > len(cprofiles):
                raise RuntimeError(
                    'The width is not enough to contain the channels with'
                    ' the given pixel size.')

            channels_corr = np.correlate(cprofiles, channels_locator)


            # Remove the effect from the padding
            ones_profiles = np.zeros(len(profiles) + 2 * offset_start)
            ones_profiles[offset_start:-offset_start] = 1

            walls_corr = (np.correlate(cprofiles, walls_locator)
                          / np.correlate(ones_profiles, walls_locator))

            arg = np.argmax(channels_corr / walls_corr)
            # from matplotlib.pyplot import figure, plot, show
            # figure()
            # plot(cprofiles / np.sum(cprofiles))
            # plot(arg + np.arange(len(mask)), mask / np.sum(mask))
            # show()

            return arg, np.max(channels_corr / walls_corr)

        # Try pixel size between 0.7 and 1.4 times what the user thinks is true
        pix_size = pix_size_init
        pix_list = np.logspace(-0.5, 0.5, 100, base=2) * pix_size
        result = []
        for pix_size in pix_list:
            try:
                result.append([pix_size, find_channels(profiles, pix_size)[1]])
            except RuntimeError:
                pass
        result = np.asarray(result)
        result[:, 1] = gfilter(result[:, 1], 3)
        pix_size = self.subpixel_find_extrema(
            result[:, 0], result[:, 1], 'max')
        left_arg, _ = find_channels(profiles, pix_size)

        centers_distance = ((self.metadata["KEY_MD_WALLWIDTH"]
                             + self.metadata["KEY_MD_WY"]) / pix_size)
        channel_width = self.metadata["KEY_MD_WY"] / pix_size
        centers = (left_arg + channel_width / 2
                   + np.arange(number_profiles) * centers_distance)

        return centers, pix_size

    def correlate_profiles(self, profiles, centers, pix_size):
        """Realign the profiles from a first approximation."""
        number_profiles = self.metadata["KEY_MD_NCHANNELS"]
        channel_width = self.metadata["KEY_MD_WY"] / pix_size
        wall_width = self.metadata["KEY_MD_WALLWIDTH"] / pix_size

        # Remove dust, bright peaks (10% median filter)
        small_filter_width = int(round(channel_width / 10))
        if small_filter_width > 3:
            profiles = median_filter(profiles, small_filter_width)

        centers_distance = np.mean(np.abs(np.diff(centers)))
        # Correct the centers by correlating the profiles
        margin = centers_distance / 2 * 1.2
        wall_margin = int(wall_width)
        extended_profiles = np.ones(len(profiles) + 2 * wall_margin)
        extended_profiles *= np.percentile(
            profiles, wall_width / channel_width)
        extended_profiles[wall_margin:-wall_margin] = profiles
        number_pixel = len(extended_profiles)
        pixels_x = np.arange(number_pixel) - wall_margin

        pos = np.zeros(len(centers) - 1)
        for i in range(len(centers) - 1):

            mask1 = np.abs(pixels_x - centers[i]) < margin
            mask2 = np.abs(pixels_x - centers[i + 1]) < 2 * margin
            p1 = extended_profiles[mask1][::-1]
            p2 = extended_profiles[mask2]
            right1 = pixels_x[number_pixel - np.argmax(mask1[::-1])]
            left2 = pixels_x[np.argmax(mask2)]
            corr = np.correlate(p2, p1)
            pos_corr = np.arange(len(corr)) + left2 + right1
            mask_expected = (
                np.abs(pos_corr - (centers[i] + centers[i+1])) < margin / 2)
            arg_max = self.subpixel_find_extrema(
                pos_corr, corr, 'max', mask_expected = mask_expected)
            pos_symmetry = arg_max / 2
            pos[i] = pos_symmetry

        centers_distance = np.mean(np.diff(pos))

        pix_size = ((self.metadata["KEY_MD_WALLWIDTH"]
                     + self.metadata["KEY_MD_WY"]) / centers_distance)

        offset = np.mean(
            pos - np.arange(number_profiles - 1) * centers_distance)

        centers = (offset - centers_distance / 2
                   + np.arange(number_profiles) * centers_distance)

        return centers, pix_size

    def subpixel_find_extrema(self, x, y, mode='min', r=2, mask_expected=None):
        """Find the maximum with subpixel accuracy, ignoring sides."""

        if mode == 'min':
            get_arg = np.argmin
            extrema_filter = minimum_filter1d
        elif mode == 'max':
            get_arg = np.argmax
            extrema_filter = maximum_filter1d
        else:
            raise RuntimeError()

        arg_extrema = np.ravel(np.where(extrema_filter(y, int(r)) == y)[0])
        # Remove extremas
        arg_extrema = arg_extrema[arg_extrema != 0]
        arg_extrema = arg_extrema[arg_extrema != len(y)]
        if mask_expected is not None:
            # Search in the expected positions first
            filter_mask = np.in1d(
                arg_extrema, np.ravel(np.argwhere(mask_expected)))
            if np.any(filter_mask):
                arg_extrema = arg_extrema[filter_mask]

        best_arg = arg_extrema[get_arg(y[arg_extrema])]

        polyfit = np.polyfit(x[best_arg - r: best_arg + r + 1],
                             y[best_arg - r: best_arg + r + 1],
                             2)
        return - polyfit[1] / (2 * polyfit[0])

    def find_center_brightwall(self, profiles):
        """Find center of brightwalls assuming a central injection."""
        number_profiles = self.metadata["KEY_MD_NCHANNELS"]

        # Filter heavely and get position of the centers as a first approx.
        filter_width = len(profiles) / ((number_profiles * 2 + 1) * 3 * 2)
        Hfiltered = gfilter(profiles, filter_width)
        maxs = np.where(maximum_filter1d(
            Hfiltered, int(filter_width)) == Hfiltered)[0]

        # Filter lightly
        soft_width = filter_width / 50
        if soft_width < 3:
            soft_width = 3
        fprof = gfilter(profiles, soft_width)

        # If max negative, not a max
        maxs = maxs[profiles[maxs] > 0]
        # If we have enough pixels
        if filter_width > 6:
            # If filter reduces int by 90%, probably a wall
            maxs = (maxs[(profiles[maxs] - fprof[maxs]) / profiles[maxs] < .5])
            # Remove sides
            maxs = maxs[np.logical_and(
                maxs > 3 / 2 * filter_width,
                maxs < len(fprof) - 3 / 2 * filter_width)]

        expected_dist = (
            self.metadata["KEY_MD_WALLWIDTH"] + self.metadata["KEY_MD_WY"]
            ) / self.metadata["KEY_MD_PIXSIZE"]
        distances = np.diff(maxs) / expected_dist
        idx = 0
        # make sure all walls and peaks are detected
        # We expect 0.5 everywhere
        while idx < len(distances):
            if distances[idx] > 0.75:
                # missed one?
                med = (maxs[idx] + maxs[idx + 1]) // 2
                maxs = np.insert(maxs, idx + 1, med)
                distances = np.diff(maxs) / expected_dist

            idx = idx + 1
        if len(maxs) == 2 * number_profiles + 1:
            maxs = maxs[1::2]
        elif len(maxs) == 2 * number_profiles:
            # One of the walls is missing, guess which one:
            left = np.polyfit(np.arange(number_profiles), maxs[1::2], 1)[1]
            right = np.polyfit(np.arange(number_profiles), maxs[::2], 1)[1]
            if left > right:
                maxs = maxs[1::2]
            else:
                maxs = maxs[::2]
        elif len(maxs) == 2 * number_profiles - 1:
            maxs = maxs[::2]

        # Sort and check number
        maxs = sorted(maxs)
        if len(maxs) != number_profiles:
            raise RuntimeError("Can't find the center of the channels!")

        return self.max_to_center(maxs)

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
        new_pixel_size = prof_width / prof_npix
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

    def extract_profiles(self, infos):
        """Extract profiles from a single scan"""
        lin_profiles = infos['Data']
        channel_width = self.metadata["KEY_MD_WY"]
        centers = infos["Centers"]
        flowdir = infos["flow direction"]
        pixel_size = infos["Pixel size"]

        prof_npix = int(np.round(channel_width / pixel_size))

        if (np.min(centers) - prof_npix / 2 < 0 or
                np.max(centers) + prof_npix / 2 > len(lin_profiles)):
            raise RuntimeError('Channel not fully contained in the image')

        profiles, pixel_size = self.interpolate_profiles(
            lin_profiles, centers, flowdir, prof_npix,
            prof_npix * pixel_size, pixel_size)

        # If inlet position unknown, guess
        if infos['Inlet location'] is None:
            if profiles[-1].max() > profiles[0].max():
                infos['Inlet location'] = 'right'
            else:
                infos['Inlet location'] = 'left'

        # If image upside down, turn
        if infos['Inlet location'] == 'right':
            profiles = profiles[::-1]
            centers = centers[::-1]
            flowdir = flowdir[::-1]

        infos["Centers"] = centers
        infos["flow direction"] = flowdir
        infos["Pixel size"] = pixel_size
        infos["Profiles noise std"] = self.get_noise(
            infos, prof_npix=prof_npix)
        return profiles

    def get_noise(self, infos, prof_npix=None):
        """get_noise"""
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        centers = infos["Centers"]
        lin_profiles = infos['Data']
        if prof_npix is None:
            pixel_size = (channel_width + wall_width) / \
                np.abs(np.mean(np.diff(centers)))
            prof_npix = int(np.round(channel_width / pixel_size))

        if self.metadata["KEY_MD_BRIGHTWALL"]:
            inmask = np.any(
                np.abs(np.arange(len(lin_profiles))[:, np.newaxis]
                       - centers[np.newaxis]) < .45 * prof_npix, axis=1)
            inmask = np.logical_and(inmask, np.isfinite(lin_profiles))
            inmask = np.logical_and(inmask, np.isfinite(infos["noise_var"]))

            noise_var = infos["noise_var"]

            filter_width = prof_npix / 100
            if filter_width < 3:
                filter_width = 3

            lbl, n = label(inmask)
            fprof = lin_profiles.copy()

            var = lin_profiles.copy()

            for i in range(1, n+1):
                mask = lbl == i
                fprof[mask] = gfilter(lin_profiles[mask], filter_width)
                var[mask] = np.square(gfilter(lin_profiles[mask] - fprof[mask],
                   filter_width)) / np.abs(fprof[mask])

            var_factor = np.median(var[inmask])

            fnoise_var = gfilter(noise_var, filter_width)
            fnoise_var[np.isnan(fnoise_var)] = np.nanmedian(fnoise_var)
            noise = var_factor * fnoise_var
            # noise[fprof < 0] = 0

        else:
            outmask = np.all(
                np.abs(np.arange(len(lin_profiles))[:, np.newaxis]
                       - centers[np.newaxis]) > .55 * prof_npix, axis=1)
            outmask = np.logical_and(outmask, np.isfinite(lin_profiles))
            outmask = np.logical_and(outmask, np.isfinite(infos["noise_var"]))
    
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
                        window = 2 * (len(background) // 2) - 1
                    wall_var[i] = np.sum(np.square(
                        background
                        - savgol_filter(background, window, window // 6)))
                lin_var_list[i] = np.sum(infos["noise_var"][mask])
    
            noise = infos["noise_var"] / np.sum(lin_var_list) * np.sum(wall_var)
            min_var = np.sum(wall_var) / np.sum(outmask)
            noise[noise < min_var] = min_var

        noise, __ = self.interpolate_profiles(
            noise, centers, infos["flow direction"],
            prof_npix, channel_width, infos["Pixel size"])
        return np.sqrt(noise)

    def align_profiles(self, infos):
        ignore = self.settings["KEY_STG_IGNORE"]
        rebin = self.settings["KEY_STG_REBIN"]
        pixel_size = infos["Pixel size"]
        centers = infos["Centers"]
        flowdir = infos["flow direction"]
        channel_width = self.metadata["KEY_MD_WY"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        profiles = infos['Profiles']
        lin_profiles = infos['Data']

        prof_npix = np.shape(profiles)[1] * rebin

        infos_tmp = dp.size_profiles(infos.copy(), self.metadata, self.settings)
        fits = infos_tmp['Fitted Profiles']

        profile_slice = dp.ignore_slice(ignore, pixel_size)

        new_centers = np.array(centers, dtype=float)
        for i, (cent, fd) in enumerate(zip(centers, flowdir)):
            p1 = profiles[i, profile_slice]
            p2 = fits[i, profile_slice]

            # Get diff between profile and fit
            diff = (dp.center(
                np.correlate(p1 - np.mean(p1), p2 - np.mean(p2), mode='full'))
                - len(profiles[i, profile_slice]) + 1)

            if not np.isnan(diff):
                diff *= rebin
                switch = self.should_switch(fd)

                if switch:
                    diff *= -1

                new_centers[i] = cent + diff

        meandist = np.mean(np.diff((new_centers[1:] + new_centers[:-1]) / 2))
        pixel_size = np.abs((channel_width + wall_width) / meandist)

        res = (new_centers - np.min(new_centers) + meandist / 2) % meandist
        offset = (np.mean(res[::2]) - np.mean(res[1::2])) / 2

        new_centers -= offset * (-1)**np.arange(len(profiles))

        new_profiles, pixel_size = self.interpolate_profiles(
            lin_profiles, new_centers, flowdir,
            prof_npix, channel_width, pixel_size)

        infos["Centers"] = new_centers
        infos["Pixel size"] = pixel_size
        infos["Profiles noise std"] = self.get_noise(
            infos, prof_npix=prof_npix)
        infos['Profiles'] = new_profiles
        return infos
