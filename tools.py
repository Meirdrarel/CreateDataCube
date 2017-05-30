import math
import numpy as np
from astropy.io import fits
import os
import sys


def sky_coord_to_galactic(xcen, ycen, pos_angl, incl, im_size=(240, 240)):
    """
    Convert position from Sky coordinates to Galactic coordinates

    :param float xcen: position of the center in arcsec
    :param float ycen: position of the center in arcsec
    :param float pos_angl: position angle of the major axis degree
    :param float incl: inclination of the disk in degree
    :param ndarray im_size: maximum radius of the scene (arcsec),
                          im_size should be larger than the slit length + seeing (Default im_size=100)
    :param float res: resolution of the high resolution data (arcsec),
                      res should be at least n x pixel size (Default res=0.04)
    :return ndarray: [r, theta]
    """
    y, x = np.indices(im_size)
    den = (y - ycen) * math.cos(math.radians(pos_angl)) - (x - xcen) * math.sin(math.radians(pos_angl))
    num = - (x - xcen) * math.cos(math.radians(pos_angl)) - (y - ycen) * math.sin(math.radians(pos_angl))
    r = (den ** 2 + (num / math.cos(math.radians(incl))) ** 2) ** 0.5
    tpsi = num * 1.

    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = math.cos(math.radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # signe
    ctheta = sg * (math.cos(math.radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane
    
    return [r, ctheta]


def rebin_data(data, new_bin):
    """
    Rebin an image.

    :param ndarray data: array to rebin
    :param int new_bin: size of the new bin
    """
    if data.ndim == 2:
        data2 = data.reshape(int(data.shape[0] / new_bin), new_bin, int(data.shape[1] / new_bin), new_bin)
        # return np.mean(data2, axis=(1, 3))
        return data2.mean(1).mean(2)

    if data.ndim == 3:
        data2 = data.reshape(data.shape[0], int(data.shape[1] / new_bin), new_bin, int(data.shape[2] / new_bin), new_bin)
        return np.mean(data2, axis=(2, 4))


def write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, rd, sig0, data, filename, oversample=1, chi2r=None, dof=None, mask=None):

    if mask is not None:
        data[np.logical_not(mask)] = float('nan')

    hdu = fits.PrimaryHDU(data=data)
    hdu.header.append(('PA', pos_angl, 'position angle in degree'))
    hdu.header.append(('INCL', incl, 'inclination in degree'))
    hdu.header.append(('XCEN', xcen / oversample, 'center abscissa in pixel'))
    hdu.header.append(('YCEN', ycen / oversample, 'center ordinate in pixel'))
    hdu.header.append(('RD', rd / oversample, 'characteristic radius in pixel'))
    hdu.header.append(('MAX_VEL', vmax, 'maximum velocity in km/s'))
    hdu.header.append(('SYST_VEL', syst_vel, 'systemic velocity in km/s'))
    hdu.header.append(('SIG0', sig0, 'dispersion velocity in km/s'))
    if chi2r:
        hdu.header.append(('CHI2R', chi2r, 'reduced chi square'))
        hdu.header.append(('DOF', dof, 'degree of freedom'))

    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename + '.fits', checksum=True, overwrite=True)


def search_file(path, filename):
    while True:
        file_list = os.listdir(path)
        if filename in file_list:
            return path+filename
        else:
            print('File {} not found in directory {}'.format(filename, path))
            sys.exit()
