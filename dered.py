'''
Mostly a copypasta/transliteration of the IDL library ccm_dered.pro routine.
Dereddens a flux vector according to the CCM 1989 parameterization.

Transliterator: Isaac Shivvers, Oct. 2013

Original documentation:
;+
; NAME:
;     CCM_UNRED
; PURPOSE:
;     Deredden a flux vector using the CCM 1989 parameterization 
; EXPLANATION:
;     The reddening curve is that of Cardelli, Clayton, and Mathis (1989 ApJ.
;     345, 245), including the update for the near-UV given by O'Donnell 
;     (1994, ApJ, 422, 158).   Parameterization is valid from the IR to the 
;     far-UV (3.5 microns to 0.1 microns).    
;
;     Users might wish to consider using the alternate procedure FM_UNRED
;     which uses the extinction curve of Fitzpatrick (1999).
; CALLING SEQUENCE:
;     CCM_UNRED, wave, flux, ebv, funred, [ R_V = ]      
;             or 
;     CCM_UNRED, wave, flux, ebv, [ R_V = ]      
; INPUT:
;     WAVE - wavelength vector (Angstroms)
;     FLUX - calibrated flux vector, same number of elements as WAVE
;             If only 3 parameters are supplied, then this vector will
;             updated on output to contain the dereddened flux.
;     EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
;             then fluxes will be reddened rather than deredenned.
;
; OUTPUT:
;     FUNRED - unreddened flux vector, same units and number of elements
;             as FLUX
;
; OPTIONAL INPUT KEYWORD
;     R_V - scalar specifying the ratio of total selective extinction
;             R(V) = A(V) / E(B - V).    If not specified, then R_V = 3.1
;             Extreme values of R(V) range from 2.75 to 5.3
;
; EXAMPLE:
;     Determine how a flat spectrum (in wavelength) between 1200 A and 3200 A
;     is altered by a reddening of E(B-V) = 0.1.   Assume an "average"
;     reddening for the diffuse interstellar medium (R(V) = 3.1)
;
;       IDL> w = 1200 + findgen(40)*50      ;Create a wavelength vector
;       IDL> f = w*0 + 1                    ;Create a "flat" flux vector
;       IDL> ccm_unred, w, f, -0.1, fnew  ;Redden (negative E(B-V)) flux vector
;       IDL> plot,w,fnew                   
;
; NOTES:
;     (1) The CCM curve shows good agreement with the Savage & Mathis (1979)
;             ultraviolet curve shortward of 1400 A, but is probably
;             preferable between 1200 and 1400 A.
;     (2)  Many sightlines with peculiar ultraviolet interstellar extinction 
;             can be represented with a CCM curve, if the proper value of 
;             R(V) is supplied.
;     (3)  Curve is extrapolated between 912 and 1000 A as suggested by
;             Longo et al. (1989, ApJ, 339,474)
;     (4) Use the 4 parameter calling sequence if you wish to save the 
;               original flux vector.
;     (5) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
;             curve (3.3 -- 8.0 um-1).    But since their revised curve does
;             not connect smoothly with longer and shorter wavelengths, it is
;             not included here.
;
; REVISION HISTORY:
;       Written   W. Landsman        Hughes/STX   January, 1992
;       Extrapolate curve for wavelengths between 900 and 1000 A   Dec. 1993
;       Use updated coefficients for near-UV from O'Donnell   Feb 1994
;       Allow 3 parameter calling sequence      April 1998
;       Converted to IDLV5.0                    April 1998
;-
'''
import numpy as np

try:
    # pyfits recent moved to subset of astropy package
    import pyfits as pf
except:
    from astropy.io import fits as pf

from ephem._libastro import eq_gal
from urllib2 import urlopen
import xml.etree.ElementTree as xmltree
import re
from astropysics import obstools

def load_dust_maps(dust_map_location='/home/isaac/Working/observations/dust_maps/'):
    hdu = pf.open(dust_map_location+'SFD_dust_4096_ngp.fits')[0]
    nsgp_n = hdu.header['LAM_NSGP']
    scale_n = hdu.header['LAM_SCAL']
    map_n = hdu.data
    hdu = pf.open(dust_map_location+'SFD_dust_4096_sgp.fits')[0]
    nsgp_s = hdu.header['LAM_NSGP']
    scale_s = hdu.header['LAM_SCAL']
    map_s = hdu.data
    MAP_DICT = {}
    MAP_DICT['N'] = [map_n, nsgp_n, scale_n]
    MAP_DICT['S'] = [map_s, nsgp_s, scale_s]
    return MAP_DICT
## loading the physical dust maps has been rendered obsolete
## by the updated function get_galactic_reddening
# try:
#     MAP_DICT = load_dust_maps()
# except:
#     print 'WARNING: cannot find/open dust maps, some functions may not work.'


def dered_CCM(wave, flux, EBV, R_V=3.1):
    '''
    Deredden a spectrum according to the CCM89 Law.
    wave: 1D array (Angstroms)
    flux: 1D array (whatever units)
    EBV: E(B-V)
    R_V: Reddening coefficient to use (default 3.1)
    '''
    x = 10000./ wave  #Convert to inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ## Infrared ##
    mask = (x > 0.3) & (x < 1.1)
    if np.any(mask):
        a[mask] =  0.574 * x[mask]**(1.61)
        b[mask] = -0.527 * x[mask]**(1.61)

    ## Optical/NIR ##
    mask = (x >= 1.1) & (x < 3.3)
    if np.any(mask):
        xxx = x[mask] - 1.82
        # c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085, #Original
        #        0.01979, -0.77530,  0.32999 ]               #coefficients
        # c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434, #from CCM89
        #       -0.62251,  5.30260, -2.09002 ]
        c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137,     #New coefficients
              -1.718,   -0.827,    1.647, -0.505 ]         #from O'Donnell
        c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985,     #(1994)
               11.102,    5.491,  -10.805,  3.347 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    ## Mid-UV ##
    mask = (x >= 3.3) & (x < 8.0)
    if np.any(mask):
        F_a = np.zeros_like(x[mask])
        F_b = np.zeros_like(x[mask])
        mask1 = x[mask] > 5.9
        if np.any(mask1):
            xxx = x[mask][mask1] - 5.9
            F_a[mask1] = -0.04473 * xxx**2 - 0.009779 * xxx**3
        a[mask] = 1.752 - 0.316*x[mask] - (0.104 / ( (x[mask]-4.67)**2 + 0.341 )) + F_a
        b[mask] = -3.090 + 1.825*x[mask] + (1.206 / ( (x[mask]-4.62)**2 + 0.263 )) + F_b

    ## Far-UV ##
    mask = (x >= 8.0) & (x < 11.0)
    if np.any(mask):
        xxx = x[mask] - 8.0
        c1 = [ -1.073, -0.628,  0.137, -0.070 ]
        c2 = [ 13.670,  4.257, -0.420,  0.374 ]
        a[mask] = np.poly1d(c1[::-1])(xxx)
        b[mask] = np.poly1d(c2[::-1])(xxx)

    #Now apply extinction correction to input flux vector
    A_V = R_V * EBV
    A_lambda = A_V * (a + b/R_V)
    return flux * 10.**(0.4*A_lambda)


def get_galactic_reddening( srchstr ):
    """
    Queries IRSA for galactic reddening from Schlafly & Finkbeiner (2011).
    <srchstr> must be resolvable by NED or SIMBAD; can be an 
     object name or RA,Dec in reasonable string form.
    """
    irsa_uri = "http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr=%s"
    result = urlopen( irsa_uri % srchstr.replace(' ','%20') ).read()

    EBV = None
    try:
        tree = xmltree.fromstring( result )
        if tree.attrib['status'] != 'ok':
            print 'Query failed'
            return None
        for c in tree.getchildren():
            if c.tag != 'result':
                continue
            gc = c.getchildren()
            if 'Reddening' in gc[0].text:
                # we found our spot!
                res = gc[2].find('meanValueSandF').text
                try:
                    EBV = float(re.search('\d+\.\d+', res).group())
                except:
                    continue
    except:
        pass
    return EBV



def remove_galactic_reddening( srchstr, wave, flux, R_V=3.1, verbose=False ):
    '''
    Deredden a spectrum by the Milky Way reddening due to 
     dust absorption (measured in the Schlafly & Finkbeiner 2011 maps)
    srchstr: must be resolvable by NED or SIMBAD; can be an 
     object name or RA,Dec in reasonable string form.
    wave: 1D wavelength vector (angstroms)
    flux: 1D flux vector to get dereddened
    R_V: the reddening coefficient to use
    '''
    
    EBV = get_galactic_reddening( srchstr )
    if verbose: print 'dereddening by E(B-V) =',EBV
    # return the de-reddened flux vector and the E(B-V) used
    return dered_CCM( wave, flux, EBV, R_V ), EBV


"""
Below taken from Pat!
"""

def cardelli(lam, R_v=3.1):
    """
    Calculate Alam/Av for the Cardelli dust law given a wavelength lam in angstroms
    """
    C = obstools.CardelliExtinction(Rv=R_v)
    return C.Alambda(lam)/C.Alambda('V')

def calzetti(lam):
    """
    Calculate Alam/Av for the Calzetti dust law given a wavelength lam in angstroms
    """
    C = obstools.CalzettiExtinction()
    return C.Alambda(lam)/C.Alambda('V')

def calc_extinction_from_balmer_decrement(halpha,hbeta,halpha_err=None,hbeta_err=None,recom_ratio=2.86,R_v=3.1,which='cardelli'):
    import math
    if which == 'cardelli':
        halpha_ext_ratio = cardelli(6562.81, R_v)
        hbeta_ext_ratio = cardelli(4861.0, R_v)
    elif which == 'calzetti':
        halpha_ext_ratio = calzetti(6562.81)
        hbeta_ext_ratio = calzetti(4861.0)
    else:
        raise Exception('Do not understand which law to use.')
    num = recom_ratio * hbeta / halpha
    try:
        A_v = 2.5 * math.log10(num) / (halpha_ext_ratio - hbeta_ext_ratio)    
    except:
        A_v = -99
    if (halpha_err != None) and (hbeta_err != None):
        dA_v = abs(2.5 / math.log(10) / (halpha_ext_ratio - hbeta_ext_ratio) * ( (halpha_err/halpha)**2. + (hbeta_err/hbeta)**2.) )**0.5
    else:
        dA_v = None
    return A_v, dA_v