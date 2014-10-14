"""
A quick script that converts a fits file (1d spectrum with appropriate linear
    parameters in the header) into a .flm ascii file.
"""
import pyfits as pf


def fits2flm( fitsfile, outfile ):
    """
    A quick script that converts a fits file (1d spectrum with appropriate linear
    parameters in the header) into a .flm ascii file.
    """
    h = pf.open( fitsfile )
    assert( h[0].header['ctype1'] == 'LINEAR' )
    wl0 = h[0].header['CRVAL1']
    dwl = h[0].header['CD1_1']
    wl = wl0 + np.arange( len(h[0].data) )*dwl
    fl = h[0].data

    outf = open(outfile, 'w')
    header = '# Converted from %s\n' %fitsfile +\
             '# wl (A)     fl (flam) \n'
    outf.write( header )
    if np.max( fl ) < 0.1:
        line = '%.6f    %.6e\n'
    else:
        line = '%.6f    %.6f\n'
    for i in range(len(wl)):
        outf.write( line%(wl[i],fl[i]) )
    outf.close()
    print 'saved %s to ascii format in %s.' %(fitsfile,outfile)