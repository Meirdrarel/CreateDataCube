#!/usr/bin/env python3

import os
import flux_model as fm
import velocity_model as vm
import tools
from Model3D import Model3D
from Clumps import Clumps
import argparse
import numpy as np
from astropy.io import ascii


def main(parser):

    ##################################################################
    # MODELS DICTIONARIES
    # if you want add more models, add them in the file corresponding (flux ou velocity) and add an entry in the corresponding dictionary below
    fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}
    vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}
    ##################################################################

    parser.add_argument('path', help='directory where will create cube', type=str)
    parser.add_argument('incl', help='inclination', type=int)
    parser.add_argument('vmax', help='velocity of the model', type=int)
    parser.add_argument('-fm', default='flat', help='flux model', type=str)
    parser.add_argument('-vm', default='flat', help='velocity model', type=str)
    parser.add_argument('-clump', nargs='+', dest='ifclump', help="create clump", type=float)
    parser.add_argument('-nocube', action='store_false', dest='ifcube', help="do not create cube", default=True)
    parser.add_argument('-rdf', default=3, type=float, help="characteristic radius of the flux model, by default is 3 pixels")
    parser.add_argument('-rdv', default=3, type=float, help="characteristic radius of the velocity model, by default is 3 pixels")
    parser.add_argument('-rt', default=8, type=int, help="truncated radius after which flux is set to 0")
    parser.add_argument('-slope', '--slope', dest="slope", type=float, default=0., help="slope of the dispersion, default is 0")
    parser.add_argument('-sig0', type=float, default=40, help="Velocity dispersion (or line broadening) in km/s")
    parser.add_argument('-WSD', default=False, action='store_true', help="create cube in lower resolution too")
    parser.add_argument('-size', default=(150, 150), nargs='+', type=int,
                        help="size of the cube in spaces dimensions, default is 150*150 pixels for a size of 0.04 and 30*30 for 0.2")
    parser.add_argument('-xcen', type=float, help="position of the center on abscissa, default is at the center of the image")
    parser.add_argument('-ycen', type=float, help="position of the center on ordinate, default is at the center of the image")
    parser.add_argument('-pa', default=0, type=float, help="position angle in degree, delfault is 0")
    parser.add_argument('-vs', default=0, type=float, help="systemic velocity in km/s, default is 0")
    args = parser.parse_args()

    if os.path.isdir(args.path) is False:
        os.makedirs(args.path)

    config = ascii.read('config')

    size = np.array(args.size)

    if args.WSD:
        pix_size, pix_size_ld = config['pix_size']
        over = int(pix_size_ld/pix_size)
        if args.xcen:
            xcen = (args.xcen + 0.5) * over - 0.5
        else:
            xcen = (int(np.ceil(size[1]/2/over)) + 0.5) * over - 0.5

        if args.ycen:
            ycen = (args.ycen + 0.5) * over - 0.5
        else:
            ycen = (int(np.ceil(size[0]/2/over)) + 0.5) * over - 0.5

    else:
        over = 1
        pix_size = config['pix_size'][0]
        if args.xcen:
            xcen = args.xcen
        else:
            xcen = int(np.ceil(size[1]/2))
        if args.ycen:
            ycen = args.ycen
        else:
            ycen = int(np.ceil(size[0]/2))

    rdf = args.rdf * over
    rdv = args.rdv * over
    rtrunc = args.rt * over

    if args.ifcube:
        print(pix_size)
        print('\nCreate Cube centered in {}A with a sampling of {}A and a width of {}A'.format(config['lbda0'][0],config['deltal'][0], config['lrange'][0]))

        model = Model3D(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdf, rdv, rtrunc, args.sig0, fm_list[args.fm], config['lbda0'][0], config['deltal'][0],
                        config['lrange'][0], pix_size, im_size=size, slope=args.slope)
        model.create_cube(vm_list[args.vm])
        cube_conv_HD = model.conv_psf(model.cube, config['fwhm_psf'][0]/config['pix_size'][0])
        print('\nSpectral convolution with a fwhm of {} pixels'.format(config['fwhm_lsf'][0]/config['deltal'][0]))
        cube_conv_SP = model.conv_lsf(cube_conv_HD, config['fwhm_lsf'][0]/config['deltal'][0])
        model.write_fits(cube_conv_SP, args.path+'CUBE')
        tools.write_fits(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdv, rdf, args.sig0, np.sum(cube_conv_SP, axis=0), args.path+'MAP_flux')
        tools.write_fits(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdv, rdf, args.sig0, model.v, args.path+'MAP_vel')
        tools.write_txt(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdv, rdf, args.sig0, config['fwhm_psf'][0]/config['pix_size'][0],
                        0, args.path+'param_model.txt')

        if args.WSD:
            cube_conv = model.conv_psf(cube_conv_SP, config['fwhm_psf'][1]/config['pix_size'][1]*over)
            cube_rebin = tools.rebin_data(cube_conv, over)
            model.write_fits(cube_rebin, args.path+'CUBE_LD', oversample=over)
            modv_ld = tools.rebin_data(model.v, over)
            XtoWrite = (xcen + 0.5)/over - 0.5
            YtoWrite = (ycen + 0.5)/over - 0.5
            tools.write_fits(XtoWrite, YtoWrite, args.pa, args.incl, args.vs, args.vmax, rdv, rdf, args.sig0, modv_ld, args.path+'MAP_vel_LD')
            tools.write_txt(XtoWrite, YtoWrite, args.pa, args.incl, args.vs, args.vmax, rdv/over, rdf/over, args.sig0, config['fwhm_psf'][1]/config[
                'pix_size'][1],
                            config['fwhm_lsf'][1]/config['deltal'][1], args.path+'param_model.txt')

    if args.ifclump:
        print('\nCreate Clumps')
        clump = Clumps(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdv, rtrunc, args.sig0, config['lbda0'][0], config['deltal'][0],
                       config['lrange'][0], pix_size, im_size=size, slope=args.slope)
        clump.create_clumps(args.ifclump, vm_list[args.vm], config['fwhm_lsf'][0]/config['deltal'][0])
        clump.write_fits(clump.cube, args.path + 'CLUMPS')

        if args.WSD:
            clump_conv = clump.conv_psf(clump.cube, config['fwhm_psf'][1]/config['pix_size'][1]*over)
            clump_rebin = tools.rebin_data(clump_conv, over)
            clump.write_fits(clump_rebin, args.path+'CLUMP_LD', oversample=over)

    if args.ifcube and args.ifclump:
        print('\nAdd clumps to the cube')
        cube_conv_SP += clump.cube
        model.write_fits(cube_conv_SP, args.path+'CUBE_wc', verbose=False)
        tools.write_fits(xcen, ycen, args.pa, args.incl, args.vs, args.vmax, rdv, rdf, args.sig0, np.sum(cube_conv_SP, axis=0), args.path + 'MAP_flux_wc')

        cube_rebin += clump_rebin
        model.write_fits(cube_rebin, args.path + 'CUBE_wc_LD', oversample=over, verbose=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data CUBE of one galaxy and/or clumps"
                                                 "\nCube are created with le size and the resolution needed."
                                                 "\nYou can create cube with smaller resolution with argument -WSD but all paramters must be in "
                                                 "\nthe lower resolution"
                                                 "\nMore information about the project on https://github.com/Meirdrarel/CreateDataCube",
                                     formatter_class=argparse.RawTextHelpFormatter)
    main(parser)




