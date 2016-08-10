"""
A quick script that converts a fits file (1d spectrum with appropriate linear
    parameters in the header) into a .flm ascii file.
"""
try:
    # pyfits recent moved to subset of astropy package
    import pyfits as pf
except:
    from astropy.io import fits as pf
import numpy as np


def fits2flm( fitsfile, outfile ):
    """
    A quick script that converts a fits file (1d spectrum with appropriate linear
    parameters in the header) into a .flm ascii file.
    """
    h = pf.open( fitsfile )
    assert( h[0].header['ctype1'] == 'LINEAR' )
    wl0 = h[0].header['CRVAL1']
    try:
        dwl = h[0].header['CD1_1']
    except:
        # sometimes a different keyword is used here
        dwl = h[0].header['CDELT1']
    if len(h[0].data.shape) == 1:
        wl = wl0 + np.arange( len(h[0].data) )*dwl
    else:
        wl = wl0 + np.arange( len(h[0].data[0]) )*dwl
    if len(h[0].data.shape) == 1:
        fl = h[0].data
        er = None
    else:
        # assume first array is fl, and second is er
        fl = h[0].data[0]
        er = h[0].data[1]

    outf = open(outfile, 'w')
    header = '# Converted from %s\n' %fitsfile +\
             '# wl (A)     fl (flam)   er (flam, opt)\n'
    outf.write( header )
    if np.max( fl ) < 0.1:
        if er == None:
            line = '%.6f    %.6e\n'
        else:
            line = '%.6f    %.6e    %.6e\n'
    else:
        if er == None:
            line = '%.6f    %.6f\n'
        else:
            line = '%.6f    %.6f    %.6f\n'
    for i in range(len(wl)):
        if er == None:
            outf.write( line%(wl[i],fl[i]) )
        else:
            outf.write( line%(wl[i],fl[i],er[i]) )
    outf.close()
    print 'saved %s to ascii format in %s.' %(fitsfile,outfile)