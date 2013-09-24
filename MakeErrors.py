#/usr/bin/env python

'''
This is a python script designed to produce an estimate
of the error spectrum for an optical spectrum.
This is done by subtracting a gaussian-smoothed profile
to get the noise alone, and then performing a rolling-window
standard-deviation calculation on the noise.
This rolling-window standard deviation is the returned
noise spectrum.

Note: all values are calculated per N values, NOT per x-coordinate.
 In other words, this function assumes that the spectrum is sampled on
 a regular wavelength grid.

Can be run as a command-line utility on a flipper-formatted spectrum file,
 where first two columns are wavelength(A) and flux(Flam), or imported
 and run as a function on an array.
'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, median_filter, percentile_filter

def rolling_window(a, window):
    # quick way to produce rolling windows of <window> size in numpy
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    middle = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    # now fill in the rest, reflecting about the ends
    out = np.empty( (a.shape[-1], shape[-1]) )
    begin = int( window/2 )
    end   = int( window-1-window/2 )
    out[:begin] = middle[0]
    out[begin:-end] = middle
    out[-end:] = middle[-1]
    return out

def rolling_std(a, window):
    # returns a rolling-window STD
    return np.std( rolling_window(a,window), axis=1 )
    
def cleanMe(spec, window=75, p=10):
    # returns a cleaned spectrum, removing extreme outlier peaks and troughs
    a = np.array(spec)
    topcut = percentile_filter( a, 100-p, size=window )
    a[ a>topcut ] = topcut[ a>topcut ]
    botcut = percentile_filter( a, p, size=window )
    a[ a<botcut ] = botcut[ a<botcut ]
    return a

def errSpec(spec, window=25):
    # Returns an error spectrum of optical spectrum a (1D array)
    #  Uses window size to determine smoothing kernel and rolling window size
    a = cleanMe(np.array(spec))
    noise = a - gaussian_filter1d( a, window )
    errors = rolling_std( noise, window )
    return errors


if __name__ == '__main__':
    # add error spectrum to flipper-formatted flm file, replaces
    #  any already-existant error spectrum
    import sys
    try:
        fin = sys.argv[1]
    except:
        raise Error('Usage: MakeErrors.py <filename>')
    d = np.loadtxt(fin)
    wl,fl = d[:,0], d[:,1]
    err = errSpec( fl )
    fout = open(fin, 'w')
    fout.write('#Error spectrum calculated by MakeErrors.py\n')
    for i,lam in enumerate(wl):
        fout.write('%.2f  %.8f  %.8f\n' %(lam, fl[i], err[i]) )
    fout.close()