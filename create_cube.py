#!usr/bin/env python
# import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import flux_model as fm
import velocity_model as vm
import tools
from Model3D import Model3D
from Clumps import Clumps
import argparse
import numpy as np
from astropy.io import ascii


def main(parser):
    parser.add_argument('path', help='directory where will create cube', type=str)
    parser.add_argument('incl', help='inclination', type=int)
    parser.add_argument('vmax', help='velocity of the model', type=int)
    parser.add_argument('-fm', default='flat', help='flux model', type=str)
    parser.add_argument('-vm', default='flat', help='velocity model', type=str)
    parser.add_argument('-clump', nargs='+', dest='ifclump', help='create clump', type=int)
    parser.add_argument('-nocube', action='store_false', dest='ifcube', help='create cube', default=True)
    parser.add_argument('-rdf', default=3, type=float, help='characteristic radius of the flux model, by default is 3 pixels')
    parser.add_argument('-rdv', default=3, type=float, help='characteristic radius of the velocity model, by default is 3 pixels')
    parser.add_argument('-rt', default=8, type=int, help='truncated radius after which flux is set to 0')
    parser.add_argument('-slope', '--slope', dest="slope", type=float, default=0., help="slope of the dispersion, default is 0")
    parser.add_argument('-sig0', type=float, default=40, help='Velocity dispersion (or line broadening) in km/s')
    # parser.add_argument('-HDO', default=False, action='store_true', help='create cube in high resolution only')
    parser.add_argument('-size', default=(30, 30), nargs='+', type=int, help='size of the cube in spaces dimensions')
    args = parser.parse_args()

    if os.path.isdir(args.path) is False:
        os.makedirs(args.path)

    # parameters for low resolution (typically MUSE)
    pix_size_ld = 0.2  # arcsec
    deltal_ld = 1.25   # angstrom
    lbda0 = 6562.78    # angstrom
    fwhm_psf_ld = 0.7  # arcsec ==> 0.7/0.2 = 3.5 pixels
    fwhm_lsf_ld = 2.5  # angstrom  == 2.5>/1.25 = 2 elements

    # parameters for high resolution (typically HST)
    pix_size_hd = 0.04  # arcsec
    deltal_hd = 1.25    # angstrom
    fwhm_psf_hd = 0.08  # arcsec
    fwhm_lsf_hd = 2.5   # angstrom

    # parameters for model
    smooth = 0
    # rd = args.rd      # pixels
    vmax = args.vmax   # km/s
    pa = 0       # degree
    incl = args.incl    # degree
    vs = 0       # km/s
    xcen = 15  # pixels
    ycen = 15  # pixels
    sig0 = 40    # km/s
    lrange = 50  # angstrom
    rtrunc = args.rt
    size = np.array(args.size)
    fwhm = fwhm_psf_ld/pix_size_ld  # correspond to 3.5 pixels

    ##################################################################
    # MODELS DICTIONARIES
    # if you want add more models, add them in the file corresponding (flux ou velocity) and add an entry in the corresponding dictionary below
    fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}
    vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}
    ##################################################################

    over = int(pix_size_ld/pix_size_hd)

    new_xcen = (xcen + 0.5) * over - 0.5
    new_ycen = (ycen + 0.5) * over - 0.5
    new_rdf = args.rdf * over
    new_rdv = args.rdv * over
    new_rtrunc = rtrunc * over

    if args.ifcube:
        print('\nCreate Cube')
        # for a field of 4"x4" => 20x20 pixels in muse and 100x100 pixels with hst

        model = Model3D(new_xcen, new_ycen, pa, incl, vs, vmax, new_rdf, new_rdv, new_rtrunc, sig0, fm_list[args.fm], lbda0, deltal_ld, lrange, pix_size_hd,
                        im_size=size, slope=0)
        model.create_cube(vm_list[args.vm])

        # cube_conv_smooth = model.conv_psf(model.cube, fwhm_psf_hd/pix_size_hd+2)   # 2 pixels
        cube_conv_SP = model.conv_lsf(model.cube, fwhm_lsf_ld/deltal_ld)

        tools.write_fits(new_xcen, new_ycen, pa, incl, vs, vmax, new_rdv, sig0, np.sum(cube_conv_SP, axis=0), args.path+'CUBE_flux_hd', oversample=5)
        model.write_fits(cube_conv_SP, args.path+'CUBE')
        tools.write_fits(new_xcen, new_ycen, pa, incl, vs, vmax, new_rdv, sig0, model.v, args.path+'CUBE_vel_map_hd')

        cube_conv = model.conv_psf(cube_conv_SP, fwhm*over)   # 3.5*5 = 17.5 HST's pixels
        cube_rebin = tools.rebin_data(cube_conv, int(pix_size_ld/pix_size_hd))
        model.write_fits(cube_rebin, args.path+'CUBE_rebin', oversample=int(pix_size_ld/pix_size_hd))

        modv_ld = tools.rebin_data(model.v, int(pix_size_ld/pix_size_hd))
        tools.write_fits(new_xcen/5, new_ycen/5, pa, incl, vs, vmax, new_rdv/5, sig0, modv_ld, args.path+'CUBE_vel_map', oversample=int(
                pix_size_ld/pix_size_hd))

    if args.ifclump:
        print('\nCreate Clumps')
        # create clump
        clump = Clumps(new_xcen, new_ycen, pa, incl, vs, vmax, new_rdv, new_rtrunc, sig0, lbda0, deltal_ld, lrange, pix_size_hd, im_size=size, slope=0)
        clump.create_clumps(args.ifclump, vm_list[args.vm], fwhm_lsf_ld/deltal_ld)
        clump.write_fits(clump.cube, args.path + 'CLUMP')

        clump_conv = clump.conv_psf(clump.cube, fwhm*5)   # 3.5*5 = 17.5 HST's pixels
        clump_rebin = tools.rebin_data(clump_conv, int(pix_size_ld/pix_size_hd))
        clump.write_fits(clump_rebin, args.path+'CLUMP_rebin', oversample=int(pix_size_ld/pix_size_hd))

    if args.ifcube and args.ifclump:
        print('\nAdd clumps to the cube')
        cube_conv_SP += clump.cube
        model.write_fits(cube_conv_SP, args.path+'CUBE_wc', verbose=False)

        cube_rebin += clump_rebin
        model.write_fits(cube_rebin, args.path + 'CUBE_wc_rebin', oversample=int(pix_size_ld / pix_size_hd), verbose=False)

        tools.write_fits(new_xcen, new_ycen, pa, incl, vs, vmax, new_rdv, sig0, np.sum(cube_conv_SP, axis=0), args.path + 'CUBE_wc_flux_hd', oversample=5)

    ascii.write(np.array([xcen, ycen, pa, incl, vs, vmax, args.rdv, sig0, fwhm, fwhm_lsf_ld, smooth]),
                args.path + 'param_model.txt',
                names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd', 'sig0', 'psfx', 'psfz', 'smooth'], format='fixed_width', delimiter=None,
                formats={'x': '%5.1f', 'y': '%5.1f', 'pa': '%5.1f', 'incl': '%5.1f', 'vs': '%5.1f', 'vm': '%5.1f', 'rd': '%5.1f', 'sig0': '%5.1f',
                         'psfx': '%5.1f', 'psfz': '%5.1f', 'smooth': '%5.1f'}, overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data CUBE of one galaxy and/or clumps"
                                                 "\nBy default this program create low and high resolution cubes, parameters have to be determined in the low "
                                                 "resolution"
                                                 "\nThe possibility to create only 'high resolution' coming ASAP",
                                     formatter_class=argparse.RawTextHelpFormatter)
    main(parser)








