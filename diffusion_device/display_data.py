# -*- coding: utf-8 -*-
"""
Functions used to display and save results of fitting

Created on Wed Aug  9 19:30:34 2017

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
import os
from matplotlib.image import NonUniformImage
import shutil
import re

from . import profile as dp


def save_plot_filt(profiles, filts, pixel_size, profiles_filter, outpath=None):
    figure()
    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    plot(X, dp.get_fax(profiles), label="data")
    plot(X, dp.get_fax(filts), label="filtered")

    plt.xlabel('Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')
    plt.title(
        "Savitzky-Golay: w{}, o{}".format(profiles_filter[0], profiles_filter[1]))
    plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_filt_fig.pdf')


def plot_single(radius, profiles, fits, lse, pixel_size,
                signal_noise, radius_error=None, prefix=''):
    # =========================================================================
    # Fit
    # =========================================================================

    if len(np.shape(radius)) > 0:
        title = (prefix + 'LSE = {:.3f}, pixel = {:.3f} um'.format(
            lse, pixel_size * 1e6))
    else:
        title = (prefix + 'r= {:.2f}$\pm${:.2f}nm, LSE = {:.3f}, '
                 'pixel = {:.3f} um'.format(
                     radius * 1e9,
                     radius_error * 3e9,
                     lse,
                     pixel_size * 1e6))
    # =========================================================================
    # Plot
    # =========================================================================
    figure()
    plt.title(title)

    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    plot(X, dp.get_fax(profiles), 'C0', label="Profiles")
    if fits is not None:
        plot(X, dp.get_fax(fits), 'C1', label="Fits")
        plt.fill_between(X, dp.get_fax(fits - signal_noise),
                         dp.get_fax(fits + signal_noise), color="C1", alpha=0.5)

    plt.xlabel('Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')
    plt.legend()


def plot_and_save(radius, profiles, fits, infos, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    radius: float or list of floats
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
    profiles: 2d list of floats
        The extracted profiles
    fits: 2d list of floats
        The fits
    pixel_size: float
        The detected pixel size
    outpath: path
        Folder where to save the figures and data

    """
    lse = infos["Reduced least square"]
    pixel_size = infos["Pixel size"]
    radius_error = infos["Radius error std"]
    signal_over_noise = infos["Signal over noise"]

    if len(np.shape(radius)) > 0:
        Rs, spectrum = radius
        figure()
        plot(Rs * 1e9, spectrum, 'x-')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(outpath + '_rSpectrum_fig.pdf')

    plot_single(radius, profiles, fits, lse, pixel_size,
                infos["Profiles noise std"], radius_error)

    # ==========================================================================
    # Save
    # ==========================================================================

    if outpath is not None:
        plt.savefig(outpath + '_fig.pdf')
        with open(outpath + '_result.txt', 'wb') as f:
            f.write("Reduced least square: {:f}\n".format(lse).encode())
            f.write("Apparent pixel size: {:f} um\n".format(pixel_size *
                                                            1e6).encode())
            f.write("Signal over noise: {:f}\n".format(
                signal_over_noise).encode())
            if len(np.shape(radius)) > 0:
                f.write("radius:\n".encode())
                np.savetxt(f, radius[0])
                f.write("Spectrum:\n".encode())
                np.savetxt(f, radius[1])
                f.write("Radius error std:\n".encode())
                np.savetxt(f, radius_error)

            else:
                f.write("Radius: {:f} nm\n".format(radius * 1e9).encode())
                f.write(
                    "Radius error std: {:f} nm\n".format(
                        radius_error * 1e9).encode())
            f.write("Profiles:\n".encode())
            np.savetxt(f, profiles)
            f.write('Fits:\n'.encode())
            np.savetxt(f, fits)


def plot_and_save_stack(radius, profiles, fits, infos,
                        settings, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    radius: list of floats
        A list of:
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
    profiles: 3d list of floats
        The extracted profiles
    fits: 3d list of floats
        The Fits
    pixel_size: list of floats
        The detected pixel size.
    images: array of floats
        The data that was analysed
    overexposed: list of bool
        For each data file, is the file overexposed?
    outpath: path
        Folder where to save the figures and data
    plotpos: array of ints
        Positions to plot if this is a stack

    """
    plotpos = settings["KEY_STG_STACK_POSPLOT"]
    overexposed = np.asarray(infos["Overexposed"])
    pixel_size = np.asarray(infos["Pixel size"])

    intensity = np.zeros(len(profiles))
    for i, p in enumerate(profiles):
        if p is None:
            intensity[i] = np.nan
        else:
            intensity[i] = np.nanmean(p)

    LSE = np.asarray(infos["Reduced least square"])
    signal_over_noise = infos["Signal over noise"]

    x = np.arange(len(radius))
    valid = np.logical_not(overexposed)
    radius_error = np.asarray(infos["Radius error std"])
    if len(np.shape(radius)) == 3:
        Rs = radius[[np.all(np.isfinite(r)) for r in radius]][0][0] * 1e9
        ylim = (0, len(radius))
        xlim = (np.min(Rs), np.max(Rs))
        figure()
        im = NonUniformImage(plt.gca(), extent=(*xlim, *ylim))
        im.set_data(Rs, np.arange(len(radius)), radius[:, 1])
        plt.gca().images.append(im)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel('Radius [nm]')
        plt.ylabel('Frame number')
    else:
        figure()
        plt.errorbar(x[valid], radius[valid] * 1e9,
                     yerr=radius_error[valid] * 3e9, fmt='x', label='data')
        plt.xlabel('Frame number')
        plt.ylabel('Radius [nm]')
        if np.any(overexposed):
            plt.errorbar(x[overexposed], radius[overexposed] * 1e9,
                         yerr=radius_error[overexposed] * 3e9, fmt='x',
                         label='overexposed data')
            plt.legend()
#        if settings["KEY_STG_RLOG"]:
#            plt.yscale("log")

    if outpath is not None:
        plt.savefig(outpath + '_R_fig.pdf')

    figure()
    plot(x[valid], LSE[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Reduced least square')
    if np.any(overexposed):
        plot(x[overexposed], LSE[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_LSE_fig.pdf')

    figure()
    plt.semilogy(x[valid], (LSE / signal_over_noise)[valid],
                 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Normalised reduced least square')
    if np.any(overexposed):
        plt.semilogy(x[overexposed], (LSE / signal_over_noise)[overexposed],
                     'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_LSE_o_SON_fig.pdf')

    figure()
    plot(x[valid], intensity[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Mean intensity')
    if np.any(overexposed):
        plot(x[overexposed], intensity[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_mean_intensity_fig.pdf')

    if len(np.shape(pixel_size)) > 0:
        figure()
        plot(x, pixel_size * 1e6, 'x')
        plt.xlabel('Frame number')
        plt.ylabel('Pixel size')
        if outpath is not None:
            plt.savefig(outpath + '_pixel_size_fig.pdf')

    if outpath is not None:
        with open(outpath + '_result.txt', 'wb') as f:
            f.write('Reduced least square:\n'.encode())
            np.savetxt(f, LSE[np.newaxis])
            if np.any(overexposed):
                f.write('Overexposed Frames:\n'.encode())
                np.savetxt(f, overexposed[np.newaxis])
            f.write('Pixel size:\n'.encode())
            np.savetxt(f, pixel_size[np.newaxis])
            f.write('Signal over noise:\n'.encode())
            np.savetxt(f, signal_over_noise[np.newaxis])

            if len(np.shape(radius)) == 3:
                f.write(f'Radii [nm]:\n'.encode())
                np.savetxt(f, Rs[np.newaxis])
                f.write(f'Spectrums:\n'.encode())
                np.savetxt(f, radius[:, 1])
                f.write('radius error:\n'.encode())
                np.savetxt(f, radius_error)

            else:
                f.write('radius:\n'.encode())
                np.savetxt(f, radius[np.newaxis])
                f.write('radius error:\n'.encode())
                np.savetxt(f, radius_error[np.newaxis])

            for i, (prof, fit) in enumerate(zip(profiles, fits)):
                if prof is not None and fit is not None:
                    f.write(f"Frame {i}\nProfiles:\n".encode())
                    np.savetxt(f, prof)
                    f.write('Fits:\n'.encode())
                    np.savetxt(f, fit)
                else:
                    f.write(f"Frame {i}\nEmpty\n".encode())

    if plotpos is not None:
        plotpos = np.asarray(plotpos)

        for pos in plotpos[plotpos < len(profiles)]:
            if profiles[pos] is None:
                continue
            pixs = pixel_size
            if len(np.shape(pixel_size)) > 0:
                pixs = pixel_size[pos]

            plot_single(radius[pos], profiles[pos], fits[pos], LSE[pos],
                        pixs, infos["Profiles noise std"][pos], radius_error[pos], prefix=f'{pos}: ')

            if outpath is not None:
                plt.savefig(outpath + '_{}_fig.pdf'.format(pos))


def prepare_output(outpath, settingsfn, metadatafn):
    """Prepare output folder

    Parameters
    ----------
    outpath: path
        Folder where to save the figures and data
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the fit settings file

    Returns
    -------
    base_name: path
        The prefix to use to save data

    """
    base_name = None
    if outpath is not None:
        settings_name = os.path.splitext(os.path.basename(settingsfn))[0]
        metadata_name = os.path.splitext(os.path.basename(metadatafn))[0]
        if re.match("metadata$", metadata_name, re.IGNORECASE):
            metadata_name = os.path.basename(os.path.dirname(metadatafn))
        if re.match(".+metadata$", metadata_name, re.IGNORECASE):
            metadata_name = metadata_name[:-8]
        if metadata_name[-1] == '_':
            metadata_name = metadata_name[:-1]
        newoutpath = os.path.join(
            outpath,
            settings_name)

        if not os.path.exists(newoutpath):
            os.makedirs(newoutpath)
        shutil.copy(
            settingsfn,
            os.path.join(
                newoutpath,
                settings_name +
                '.json'))

        base_name = os.path.join(
            newoutpath,
            metadata_name)
    return base_name
