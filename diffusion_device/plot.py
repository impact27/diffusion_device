# -*- coding: utf-8 -*-
"""
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
import tifffile

from .images import full_fit
from . import input_files
from . import profile as dp


def plotpos(settingsfn, metadatafn, outpath, plotpos=None):
    """Plot the sizing data

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file
    outpath: path
        Folder where to save the figures and data
    plotpos: array of ints
        Positions to plot if this is a stack

    """
    radius, profiles, fits, pixel_size, im, image_type, overexposed = \
        full_fit(settingsfn, metadatafn)

    outpath = prepare_output(outpath, settingsfn, metadatafn)

    if image_type == '4pos':
        return plot4pos(radius, profiles, fits, pixel_size, im, outpath)
    elif image_type == '4pos_stack':
        return plot4posstack(radius, profiles, fits, pixel_size, im,
                             overexposed, outpath, plotpos)
    elif image_type == '12pos':
        return plot12pos(radius, profiles, fits, pixel_size, im, outpath)


def plot4pos(radius, profiles, fits, pixel_size, im, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file
    outpath: path
        Folder where to save the figures and data

    """

    # =========================================================================
    # Fit
    # =========================================================================

    lse = np.sqrt(np.mean(np.square(profiles - fits)))

    if len(np.shape(radius)) > 0:
        Rs, spectrum = radius
        figure()
        plot(Rs * 1e9, spectrum, 'x-')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(outpath + '_rSpectrum_fig.pdf')
        figure()
        plt.title('LSE = {:.4e}, pixel = {:.3f} um'.format(
            lse, pixel_size * 1e6))
    else:
        figure()
        plt.title('r= {:.2f} nm, LSE = {:.4e}, pixel = {:.3f} um'.format(
            radius * 1e9, lse, pixel_size * 1e6))
    # ==========================================================================
    # Plot
    # ==========================================================================

    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    plot(X, dp.get_fax(profiles))
    plot(X, dp.get_fax(fits))

    plt.xlabel('Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')

    #==========================================================================
    # Save
    #==========================================================================

    if outpath is not None:
        tifffile.imsave(outpath + '_im.tif', im)
        plt.savefig(outpath + '_fig.pdf')
        with open(outpath + '_result.txt', 'wb') as f:
            f.write("LSE: {:e}\n".format(lse).encode())
            f.write("Apparent pixel size: {:f} um\n".format(pixel_size *
                                                            1e6).encode())
            if len(np.shape(radius)) > 0:
                f.write("radius:\n".encode())
                np.savetxt(f, radius[0])
                f.write("Spectrum:\n".encode())
                np.savetxt(f, radius[1])

            else:
                f.write("Radius: {:f} nm".format(radius * 1e9).encode())
            f.write("Profiles:".encode())
            np.savetxt(f, profiles)
            f.write('Fits:\n'.encode())
            np.savetxt(f, fits)
    return radius


def prepare_output(outpath, settingsfn, metadatafn):
    """Prepare output folder

    Parameters
    ----------
    outpath: path
        Folder where to save the figures and data
    settingsfn: path
        path to the fit settings file

    Returns
    -------
    base_name: path
        The prefix to use to save data

    """
    base_name = None
    if outpath is not None:
        newoutpath = os.path.join(
            outpath,
            os.path.splitext(os.path.basename(metadatafn))[0])
        if not os.path.exists(newoutpath):
            os.makedirs(newoutpath)
        base_name = os.path.join(
            newoutpath,
            os.path.splitext(os.path.basename(settingsfn))[0])
        shutil.copy(settingsfn, base_name + '.json')
    return base_name


def plot12pos(radius, profiles, fits, pixel_size, ims, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file
    outpath: path
        Folder where to save the figures and data

    """

    # =========================================================================
    # Plot
    # =========================================================================
    lse = np.sqrt(np.mean(np.square(profiles - fits)))
    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    if len(np.shape(radius)) > 0:
        Rs, spectrum = radius
        figure()
        plot(Rs * 1e9, spectrum, 'x-')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(outpath + '_rSpectrum_fig.pdf')
        figure()
        plt.title('LSE = {:.4e}, pixel = {:.3f} um'.format(
            lse, pixel_size * 1e6))
    else:
        figure()
        plt.title('r= {:.2f} nm, LSE = {:.4e}, pixel = {:.3f} um'.format(
            radius * 1e9, lse, pixel_size * 1e6))

    plot(X, dp.get_fax(profiles))
    plot(X, dp.get_fax(fits))
    plt.xlabel('Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')

    #==========================================================================
    # Save
    #==========================================================================

    if outpath is not None:
        tifffile.imsave(outpath + '_ims.tif', ims)
        plt.savefig(outpath + '_fig.pdf')

        with open(outpath + '_result.txt', 'wb') as f:
            if len(np.shape(radius)) > 0:
                f.write("radius:\n".encode())
                np.savetxt(f, radius[0])
                f.write("Spectrum:\n".encode())
                np.savetxt(f, radius[1])
            else:
                f.write("Radius: {:f} nm\n".format(radius * 1e9).encode())

            f.write("LSE: {:e}\n".format(lse).encode())
            f.write("Profiles:\n".encode())
            np.savetxt(f, profiles)
            f.write('Fits:\n'.encode())
            np.savetxt(f, fits)
    return radius


def plot4posstack(radius, profiles, fits, pixel_size, images, overexposed,
                  outpath=None, plotpos=None):
    """Plot the sizing data

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file
    outpath: path
        Folder where to save the figures and data
    plotpos: array of ints
        Positions to plot if this is a stack

    """
    # Infer variables

    intensity = np.asarray([np.nanmax(p) for p in profiles])
    LSE = np.zeros(len(profiles))
    for i, (p, f) in enumerate(zip(profiles, fits)):
        LSE[i] = np.sqrt(np.mean(np.square(p - f)))

    x = np.arange(len(radius))
    valid = np.logical_not(overexposed)

    if len(np.shape(radius)) == 3:
        Rs = radius[0, 0] * 1e9
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
        plot(x[valid], radius[valid] * 1e9, 'x', label='data')
        plt.xlabel('Frame number')
        plt.ylabel('Radius [nm]')
        if np.any(overexposed):
            plot(x[overexposed], radius[overexposed] * 1e9, 'x',
                 label='overexposed data')
            plt.legend()

    if outpath is not None:
        plt.savefig(outpath + '_R_fig.pdf')

    figure()
    plot(x[valid], LSE[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Least square error')
    if np.any(overexposed):
        plot(x[overexposed], LSE[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_LSE_fig.pdf')

    figure()
    plot(x[valid], intensity[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Maximum intensity')
    if np.any(overexposed):
        plot(x[overexposed], intensity[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_max_intensity_fig.pdf')

    figure()
    plot(x, pixel_size * 1e6, 'x')
    plt.xlabel('Frame number')
    plt.ylabel('Pixel size')
    if outpath is not None:
        plt.savefig(outpath + '_pixel_size_fig.pdf')

    if outpath is not None:
        tifffile.imsave(outpath + '_ims.tif', images)
        with open(outpath + '_result.txt', 'wb') as f:
            f.write('Least square error:\n'.encode())
            np.savetxt(f, LSE)
            if np.any(overexposed):
                f.write('Overexposed Frames:\n'.encode())
                np.savetxt(f, overexposed)
            f.write('Pixel size:\n'.encode())
            np.savetxt(f, pixel_size)
            if len(np.shape(radius)) == 3:
                for r, spectrum in zip(Rs, radius[:, 1]):
                    f.write('Spectrums for radius {:.4e}nm:\n'
                            .format(r).encode())
                    np.savetxt(f, spectrum)

            else:
                f.write('radius:\n'.encode())
                np.savetxt(f, radius)

    if plotpos is not None:
        plotpos = np.asarray(plotpos)

        for pos in plotpos[plotpos < len(profiles)]:

            profs = dp.get_fax(profiles[pos])
            X = np.arange(len(profs)) * pixel_size[pos] * 1e6
            figure()
            plot(X, profs)

            fits = dp.get_fax(fits[pos])

            plot(X, fits)
            if len(np.shape(radius)) == 3:
                plt.title('LSE = {:.2e}, pixel = {:.3f} um'.format(
                    LSE[pos], pixel_size[pos] * 1e6))
            else:
                plt.title(
                    'r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
                        radius[pos] * 1e9, LSE[pos], pixel_size[pos] * 1e6))
            plt.xlabel('Position [$\mu$m]')
            plt.ylabel('Normalised amplitude')

    return radius
