import numpy as np
from numpy.fft import fftshift, fft2, ifft2, irfft, rfft
from astropy.io import ascii, fits
import astropy.constants as ct
import tools


class Model3D:
    def __init__(self, xcen, ycen, pos_angl, incl, syst_vel, max_vel, rdf, rdv, rtrunc, sig0, flux_model, lbda0, dlbda, lrange, pix_size, im_size=(240, 240),
                 slope=0):
        # self.light_speed = 299792.458  # km/s
        self.center_bright = 1500 + np.random.randint(0, 1000, 1)

        self.xcen = xcen
        self.ycen = ycen
        self.pos_angl = pos_angl
        self.incl = incl
        self.syst_vel = syst_vel
        self.vmax = max_vel
        self.rdv = rdv
        self.rdf = rdf
        self.lbda0 = lbda0
        self.sig0 = sig0/ct.c.value*1e3*self.lbda0
        # self.sig0 = sig0/self.light_speed + 1
        self.slope = slope
        self.dlbda = dlbda

        self.im_size = im_size
        self.radius, self.ctheta = tools.sky_coord_to_galactic(xcen, ycen, pos_angl, incl, im_size=im_size)

        self.lsf_size = int(np.ceil(lrange / dlbda))
        self.pix_size = pix_size
        self.lrange = lrange
        self.psf = None
        self.psf_size = None
        self.psf_fft2 = None
        self.lsf = None
        self.lsf_fft = None
        self.cube = None
        self.lbda = None
        self.lbda_ind = np.arange(-self.lsf_size/2*dlbda, self.lsf_size/2*dlbda, dlbda) + self.lbda0
        self.flux = None
        self.flux_model = flux_model
        self.rtrunc = rtrunc
        self.v = None

    def disk_velocity(self, vel_model):
        """

        :param vel_model:
        :return ndarray:
        """

        vr = vel_model(self.radius, self.rdv, self.vmax)

        # Calculation of the velocity field
        v = vr * np.sin(np.radians(self.incl)) * self.ctheta + self.syst_vel

        print('\nafter projection, min vel : {:3.3f} and max vel : {:3.3f} with vs : {:3.1f}'.format(np.min(v), np.max(v), self.syst_vel))

        return v

    def create_cube(self, vel_model):

        self.flux = self.flux_model(self.xcen, self.ycen, self.pos_angl, self.incl, self.rdf, self.center_bright, self.rtrunc, self.im_size)

        v = self.disk_velocity(vel_model)
        v[np.where(self.radius > self.rtrunc)] = 0
        self.v = v

        # plt.figure()
        # plt.imshow(v, origin='lower')
        # plt.show()

        self.lbda = self.lbda0*(v/ct.c.value*1e3 + 1)
        self.cube = self.flux*np.exp(-(self.lbda-self.lbda_ind.reshape(self.lsf_size, 1, 1))**2/2/self.sig0**2)

    def conv_psf(self, data, fwhm):

        # (2*fwhm) is considered to avoid the boundary effect after the convolution
        self.psf_size = np.array([2 * fwhm, 2 * fwhm])
        ideal_size = 2 ** np.ceil(np.log(self.im_size + 2 * self.psf_size) / np.log(2))
        # to prevent any problems in the future, self.psf_size has been forced to be an array of int
        self.psf_size = np.array((ideal_size - self.im_size) / 2, dtype=int)

        y, x = np.indices((self.im_size[0] + 2 * self.psf_size[0], self.im_size[1] + 2 * self.psf_size[1]))
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.psf = (1. / (2 * np.pi * sigma ** 2)) * np.exp(-((x - self.im_size[1] / 2 - self.psf_size[1]) ** 2 + (y - self.im_size[0] / 2 - self.psf_size[
            0]) ** 2) / (2.0 * sigma ** 2))

        # normalization in order to ensure flux conservation
        self.psf /= self.psf.sum()

        self.psf_fft2 = fft2(fftshift(self.psf).reshape(1, self.psf.shape[0], self.psf.shape[1]))

        data2 = np.zeros((data.shape[0], data.shape[1] + 2 * self.psf_size[0], data.shape[2] + 2 * self.psf_size[1]))

        data2[:, self.psf_size[0]:data.shape[1] + self.psf_size[0], self.psf_size[1]:data.shape[2] + self.psf_size[1]] = data
        data_conv = ifft2(fft2(data2) * self.psf_fft2)

        return data_conv[:, self.psf_size[0]:data.shape[1] + self.psf_size[0], self.psf_size[1]:data.shape[2] + self.psf_size[1]].real

    def conv_lsf(self, data, fwhm):

        ls = int(np.ceil(2 * fwhm))
        ideal_size = 2 ** np.ceil(np.log(self.lsf_size + 2 * ls) / np.log(2))
        ls = int(np.ceil((ideal_size - self.lsf_size) / 2))

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        lindex = (np.arange(self.lsf_size + 2 * ls) - (self.lsf_size + 2 * ls) / 2)

        self.lsf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-lindex ** 2 / (2.0 * sigma ** 2))
        self.lsf /= self.lsf.sum()  # normalization in order to ensure flux conservation
        self.lsf_fft = fftshift(self.lsf)

        data2 = np.zeros((data.shape[0] + 2 * ls, data.shape[1], data.shape[2]))
        data2[ls:data.shape[0] + ls, :, :] = data

        specconv = irfft(rfft(data2, axis=0) * rfft(self.lsf_fft.reshape(self.lsf_fft.shape[0], 1, 1), axis=0), axis=0)

        specconv = specconv.real
        specconv = specconv[ls:data.shape[0] + ls, :, :]

        return specconv

    def write_fits(self, data, name, oversample=1, verbose=True):
        if oversample != 1:
            self.pix_size *= oversample
            self.xcen = (self.xcen+0.5)/oversample-0.5
            self.ycen = (self.ycen+0.5)/oversample-0.5
            self.rdv /= oversample
            self.rdf /= oversample
            self.rtrunc /= oversample
            if verbose:
                print('----------------------------------------\nfor low resolution :')
        else:
            if verbose:
                print('----------------------------------------\nfor high resolution :')
        if verbose:
            print('pixel size: {}\nxcen: {} \tycen: {}'.format(self.pix_size, self.xcen, self.ycen))
        hdu = fits.PrimaryHDU(data=data)
        hdu.header.append(('CTYPE1', 'RA--TAN', 'pixel coordinate system'))
        hdu.header.append(('CUNIT1', 'ARCSEC', 'pixel coordinate unit'))
        hdu.header.append(('CRPIX1', self.xcen, 'Reference pixel in X'))
        hdu.header.append(('CRVAL1', 0., 'Reference pixel in X'))
        hdu.header.append(('CDELT1', self.pix_size, 'Pixel size in arcsec'))
        hdu.header.append(('CTYPE2', 'DEC--TAN', 'pixel coordinate system'))
        hdu.header.append(('CUNIT2', 'ARCSEC', 'pixel coordinate unit'))
        hdu.header.append(('CRPIX2', self.ycen, 'Reference pixel in Y'))
        hdu.header.append(('CRVAL2', 0., 'Reference pixel'))
        hdu.header.append(('CDELT2', self.pix_size, 'Pixel size in arcsec'))
        hdu.header.append(('CTYPE3', 'WAVELENGTH', 'pixel coordinate system'))
        hdu.header.append(('CUNIT3', 'Angstrom', 'pixel coordinate unit'))
        hdu.header.append(('CRPIX3', 1, 'Wavelength of reference pixel'))
        hdu.header.append(('CRVAL3', self.lbda0 - self.lrange / 2, 'Reference pixel'))
        hdu.header.append(('CDELT3', self.dlbda, 'Pixel size in angstrom'))
        hdu.header.add_comment('Model Parameters')
        hdu.header.append(('XCEN', self.xcen, 'center abscissa in pixel'), end=True)
        hdu.header.append(('YCEN', self.ycen, 'center ordinate in pixel'), end=True)
        hdu.header.append(('PA', self.pos_angl, 'position angle in degree'), end=True)
        hdu.header.append(('INCL', self.incl, 'inclination in degree'), end=True)
        hdu.header.append(('RDF', self.rdf, 'characteristic radius of flux in pixel'), end=True)
        hdu.header.append(('RDV', self.rdv, 'characteristic radius of velocity in pixel'), end=True)
        hdu.header.append(('RTRUNC', self.rtrunc, 'truncated radius in pixel > flux is null'), end=True)
        hdu.header.append(('MAX_VEL', self.vmax, 'maximum velocity in km/s'), end=True)
        hdu.header.append(('SYST_VEL', self.syst_vel, 'systemic velocity in km/s'), end=True)
        hdu.header.append(('SIG0', self.sig0*ct.c.value/self.lbda0*1e3, 'dispersion velocity in km/s'), end=True)
        hdulist = fits.HDUList(hdu)
        hdulist.writeto(name + '.fits', checksum=True, overwrite=True)
        print('fits " {} "  has been written \n'.format(name))
        # print('\n----------------------------------------\n')
