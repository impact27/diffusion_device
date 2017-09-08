# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:30:34 2017

@author: quentinpeter
"""
from . import profile as dp
import matplotlib.pyplot as plt
import numpy as np
from .json import full_fit
from . import json
from matplotlib.pyplot import plot, figure
import os
from matplotlib.image import NonUniformImage
import shutil
import tifffile


def plotpos(settingsfn, metadatafn, outpath=None, plotpos=None):
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
    dtype = json.getType(metadatafn)
    if dtype == '4pos':
        return plot4pos(settingsfn, metadatafn, outpath)
    elif dtype == '4pos_stack':
        return plot4posstack(settingsfn, metadatafn, outpath, plotpos)
    elif dtype == '12pos':
        return plot12pos(settingsfn, metadatafn, outpath)


def plot4pos(settingsfn, metadatafn, outpath=None):
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

    radius, profiles, fits, lse, pixel_size, im = \
        full_fit(settingsfn, metadatafn)

    base_name = prepare_output(outpath, settingsfn, metadatafn)

    if len(np.shape(radius)) > 0:
        Rs, spectrum = radius
        figure()
        plot(Rs * 1e9, spectrum, 'x-')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(base_name + '_rSpectrum_fig.pdf')
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
        tifffile.imsave(base_name + '_im.tif', im)
        plt.savefig(base_name + '_fig.pdf')
        with open(base_name + '_result.txt', 'wb') as f:
            f.write("LSE: {:e}\n".format(lse).encode())
            f.write("Apparent pixel size: {:f} um\n".format(pixel_size *
                                                            1e6).encode())
            if len(np.shape(radius)) > 0:
                f.write("Radii:\n".encode())
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


def plot12pos(settingsfn, metadatafn, outpath=None):
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
    radius, profiles, fits, lse, pixel_size = full_fit(settingsfn, metadatafn)

    # =========================================================================
    # Plot
    # =========================================================================

    base_name = prepare_output(outpath, settingsfn, metadatafn)

    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    if len(np.shape(radius)) > 0:
        Rs, spectrum = radius
        figure()
        plot(Rs * 1e9, spectrum, 'x-')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(base_name + '_rSpectrum_fig.pdf')
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
        plt.savefig(base_name + '_fig.pdf')

        with open(base_name + '_result.txt', 'wb') as f:
            if len(np.shape(radius)) > 0:
                f.write("Radii:\n".encode())
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


def plot4posstack(settingsfn, metadatafn, outpath=None, plotpos=None):
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
    (radii, profiles_list,
     fits_list, LSE, pixs, overexposed) = full_fit(settingsfn, metadatafn)

    intensity = np.asarray([np.nanmax(p) for p in profiles_list])
    LSE = np.asarray(LSE)
    pixs = np.asarray(pixs)

    base_name = prepare_output(outpath, settingsfn, metadatafn)

    x = np.arange(len(radii))
    valid = np.logical_not(overexposed)

    if len(np.shape(radii)) == 3:
        Rs = radii[0, 0] * 1e9
        ylim = (0, len(radii))
        xlim = (np.min(Rs), np.max(Rs))
        figure()
        im = NonUniformImage(plt.gca(), extent=(*xlim, *ylim))
        im.set_data(Rs, np.arange(len(radii)), radii[:, 1])
        plt.gca().images.append(im)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel('Radius [nm]')
        plt.ylabel('Frame number')

    else:

        figure()
        plot(x[valid], radii[valid] * 1e9, 'x', label='data')
        plt.xlabel('Frame number')
        plt.ylabel('Radius [nm]')
        if np.any(overexposed):
            plot(x[overexposed], radii[overexposed] * 1e9, 'x',
                 label='overexposed data')
            plt.legend()

    if outpath is not None:
        plt.savefig(base_name + '_R_fig.pdf')

    figure()
    plot(x[valid], LSE[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Least square error')
    if np.any(overexposed):
        plot(x[overexposed], LSE[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(base_name + '_LSE_fig.pdf')

    figure()
    plot(x[valid], intensity[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Maximum intensity')
    if np.any(overexposed):
        plot(x[overexposed], intensity[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(base_name + '_max_intensity_fig.pdf')

    figure()
    plot(x, pixs * 1e6, 'x')
    plt.xlabel('Frame number')
    plt.ylabel('Pixel size')
    if outpath is not None:
        plt.savefig(base_name + '_pixel_size_fig.pdf')

    if outpath is not None:
        with open(base_name + '_result.txt', 'wb') as f:
            f.write('Least square error:\n'.encode())
            np.savetxt(f, LSE)
            if np.any(overexposed):
                f.write('Overexposed Frames:\n'.encode())
                np.savetxt(f, overexposed)
            f.write('Pixel size:\n'.encode())
            np.savetxt(f, pixs)
            if len(np.shape(radii)) == 3:
                for r, spectrum in zip(Rs, radii[:, 1]):
                    f.write('Spectrums for radius {:.4e}nm:\n'
                            .format(r).encode())
                    np.savetxt(f, spectrum)

            else:
                f.write('Radii:\n'.encode())
                np.savetxt(f, radii)

    if plotpos is not None:
        plotpos = np.asarray(plotpos)

        for pos in plotpos[plotpos < len(profiles_list)]:

            profs = dp.get_fax(profiles_list[pos])
            X = np.arange(len(profs)) * pixs[pos] * 1e6
            figure()
            plot(X, profs)

            fits = dp.get_fax(fits_list[pos])

            plot(X, fits)
            if len(np.shape(radii)) == 3:
                plt.title('LSE = {:.2e}, pixel = {:.3f} um'.format(
                    LSE[pos], pixs[pos] * 1e6))
            else:
                plt.title(
                    'r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
                        radii[pos] * 1e9, LSE[pos], pixs[pos] * 1e6))
            plt.xlabel('Position [$\mu$m]')
            plt.ylabel('Normalised amplitude')

    return radii
