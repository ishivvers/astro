#!/usr/bin/env python
'''
This is a script to turn a Flipper-group spectrum into a flux-normalized
 spectrum, given the magnitude in a certain band at the time of the spectrum.
Requires pysynphot, expects Flipper-standard .flm spectrum file, and expects
 a two-column passband file if custom passband used (wavelength in Angstroms, relative transmittance).
'''
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pysynphot as ps
from iAstro import pretty_plot_spectra

import argparse
parser = argparse.ArgumentParser(description='A script used to flux-normalize a spectrum to'+\
                                             ' a photometric observation, handling errors and headers intelligently.')
parser.add_argument('spec_file', metavar='spectrum', type=str,
                    help='path to input spectrum file')
parser.add_argument('mag', metavar='magnitude', type=float,
                    help='magnitude to which the spectrum should be renormalized')
parser.add_argument('passband', metavar='passband', type=str,
                    help='either a passband ID string or the path to a passband throughput file')
parser.add_argument('--vegamag', dest='magtype', action='store_const',
                   const='vegamag', default='abmag', help='use vegamag (default: abmag)')
parser.add_argument('--error', dest='mag_error', action='store', type=float, default=0.0,
                    help='take into account error on magnitude')
parser.add_argument('--specmult', dest='specmult', action='store', type=float, default=10**-15,
                    help='factor by which to multipy input spectrum flux')
                    # most Flipper spectrum are in units of 10**-15 flam (erg cm^-2 s^-1 A^-1)
parser.add_argument('--outfile', dest='out_file', action='store', default=None,
                    help='name of final spectrum file')
parser.add_argument('--errorfile', dest='error_file', action='store', default=None,
                    help='file containing errors, if original had errors stripped (by IRAF, for example)')
parser.add_argument('--noplot', dest='plot', action='store_false',
                    help='refrain from producing and saving plots')

args = parser.parse_args()
mag = args.mag
magtype = args.magtype
mag_error = args.mag_error
SpecMult = args.specmult
spectrum = ps.FileSpectrum( args.spec_file )
try:
    passband = ps.ObsBandpass( args.passband )
except:
    passband = ps.FileBandpass( args.passband )

if args.plot:
    # plot them up to make sure everything's copacetic
    plt.figure(1)
    tmp_spec = spectrum.flux/np.max(spectrum.flux)
    pb_norm   = passband.throughput/np.max(passband.throughput)
    fig1 = pretty_plot_spectra( spectrum.wave, tmp_spec )
    plt.title( args.spec_file.split('/')[-1] )
    plt.savefig('SpectraAndPassband.png')

# use pysynphot to renormalize the spectrum
spec_norm = spectrum.renorm( mag, magtype, passband, force=True )

# include original errors (if present)
err = False
if args.error_file:
    err = True
    flux_errors = np.loadtxt( args.error_file )
else:
    rawdat = np.loadtxt( args.spec_file )
    if rawdat.shape[-1] > 2:
        err = True
        flux_errors = rawdat[:,2]*SpecMult #assumes errors are third column of data
# find any normalization errors (if present)
if mag_error:
    # find the flux errors by normalizing to mag-mag_error 
    #  (magnitude errors are worse in positive flux direction, so this is conservative)
    spec_plus_error = spectrum.renorm( mag-mag_error, magtype, passband, force=True )
    norm_errors = np.abs( spec_plus_error.flux - spec_norm.flux )
# now produce a final error spectrum
if (err and mag_error):
    errors = (flux_errors**2 + norm_errors**2)**.5
elif (err and not mag_error):
    errors = flux_errors
elif (mag_error and not err):
    errors = norm_errors
else:
    errors = None

if args.plot:
    # plot up both, for comparision
    if err:
        fig2 = pretty_plot_spectra( spectrum.wave, spectrum.flux*SpecMult, flux_errors )
    else:
        fig2 = pretty_plot_spectra( spectrum.wave, spectrum.flux*SpecMult )
    if errors != None:
        pretty_plot_spectra( spec_norm.wave, spec_norm.flux, errors, fig=fig2 )
    else:
        pretty_plot_spectra( spec_norm.wave, spec_norm.flux, fig=fig2 )
    plt.title( args.spec_file.split('/')[-1] )
    plt.savefig('NormalizedSpectra.png')


# save the new spectrum to file

# first pull the old header out of the old file
#  and append a line about normalization to it
header = ''.join([ line for line in open( args.spec_file, 'r').readlines() if line[0]=='#' ])
header += "#Flux-normalized to {} {} in {}\n".format(mag, magtype, passband.name.split('/')[-1])
# see whether there is an error spectrum associated with original file, and keep it if there is
#  (the IRAF routines used to deredshift and deredden usually remove the error column)
#  NOTE: this assumes that the error spectrum is accurate, and does not get renormalized
if errors != None:
    header += "# wl (A)  flux (flam)  err (flam)\n"
else:
    header += "# wl (A)  flux (flam)\n"

# write header to tmpfile
open('tmp1.tmp','w').write(header)

# write data to another tmpfile
if errors != None:
    new = np.empty( [len(spec_norm.wave), 3] )
    new[:,2] = errors
    formats = ['%.2f','%.6e','%.6e']
else:
    new = np.empty( [len(spec_norm.wave), 2] )
    formats = ['%.2f','%.6e']
new[:,0] = spec_norm.wave
new[:,1] = spec_norm.flux
np.savetxt( 'tmp2.tmp', new, fmt=formats )

# smash together the two temp files into our output
if args.out_file != None:
    os.system( 'cat tmp1.tmp tmp2.tmp > {}'.format(args.out_file))
else:
    os.system( 'cat tmp1.tmp tmp2.tmp > {}.{}.renorm'.format(args.spec_file, mag))
os.system( 'rm tmp1.tmp tmp2.tmp' )

plt.show()


