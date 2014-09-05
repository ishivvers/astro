#!/usr/bin/env python

'''
This is a library of constants and functions
commonly used by me for astro work.
Everything in CGS units.

-Isaac Shivvers

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#allcolors = colors.cnames.keys()  #this allows all colors, but some are dim
allcolors = [c for c in colors.cnames.keys() if ('dark' in c) or ('medium') in c ] +\
             'r g b c m k'.split()
import re
import pyfits
from astro import dered
# looks like some versions of my python don't have the newest SciPY, so here's a hack
try:
    from scipy.integrate import trapz
    from scipy.optimize import curve_fit
    from scipy.ndimage import percentile_filter
    from scipy.interpolate import UnivariateSpline
    from scipy.ndimage import generic_filter
except:
    print 'iAstro: some scipy packages did not load; some functions may not be available'
try:
    from jdcal import gcal2jd
except:
    print 'iAstro: jdcal did not load; some functions may not be available'
try:
    import mechanize
except:
    print 'iAstro: mechanize did not load; some functions may not be available'
try:
    from astro import spectools
except:
    print 'iAstro: spectools did not load; some functions may not be available'
try:
    from scikits import datasmooth
except:
    print 'Cannot load datasmooth; smooth function will be impaired.'

class C:
    '''
    A container class that holds a bunch of constants.
    '''
    ##############################
    # constants
    ##############################
    c = 2.998E10 #cm s^-1
    sig_B = 5.67E-5 #erg cm^-2 s^-1 K^-4
    a_B = 7.56E-15 #erg cm^-3 K^-4
    k = 1.38E-16 #erg K^-1
    wein_lam = .29 #cm K^-1
    wein_nu = 5.88E10 #Hz K^-1
    h = 6.6260755E-27 #erg s
    h_bar = 1.05457266E-27 #erg s
    G = 6.674E-8 #cm^3 g^-1 s^-2
    sig_T = 6.65E-25 #cm^2
    pi = 3.141592653589793 #none
    H0_h = 3.2408E-18 #h * s^-1
    H0 = 2E-18 #s^-1
    T_cmb = 2.725 #K
    
    ##############################
    # properties
    ##############################
    m_p = 1.67E-24 #g
    m_e = 9.11E-28 #g
    M_sun = 1.988E33 #g
    M_earth = 5.972E27 #g
    R_sun = 6.955E10 #cm
    R_earth = 6.3675E8 #cm
    L_sun = 3.846E33 #erg s^-1
    e = 4.803E-10 #statC
    T_sun = 5778. # K, surface
    
    ##############################
    # conversions
    ##############################
    eV2erg = 1.602E-12 #ergs eV^-1
    year2sec = 3.154E7 #seconds yr^-1
    pc2cm = 3.086E18 #cm parsec^-1


##############################
# functions
##############################

def black_body_nu(nu, T):
    ''' blackbody curve as a function of frequency (array) and T (float) '''
    B = (2.*C.h*nu**3)/(C.c**2) * ( np.exp((C.h*nu)/(C.k*T)) - 1.)**-1
    return B

def black_body_lam(lam, T):
    ''' blackbody curve as function of wavelength (angstroms, array) and T (float) '''
    # convert lambda (in angstroms) to cm
    lam = 1E-8*lam
    B = (2.*C.h*C.c)/(lam**5) * ( np.exp((C.h*C.c)/(lam*C.k*T)) -1.)**-1
    return B

def frac_day_to_ints(frac_day, **kwargs):
    '''
    Converts a fractional day to integer hours, minutes, seconds, microseconds.
    '''
    # discards day integer value
    frac_day = frac_day%1
    hours = frac_day*24
    h = int(hours)
    minutes = (hours-h)*60
    m = int(minutes)
    seconds = (minutes-m)*60
    s = int(seconds)
    milliseconds = (seconds-s)*1000
    ms = int(milliseconds)
    return h,m,s,ms


def pretty_plot_spectra(lam, flam, err=None, label=None, multi=False, label_coord=None, fig=None, binning=None):
    '''
    Produce pretty spectra plots.
    
    lam: wavelength (A expected)
    flam: flux (ergs/cm^2/sec/A expected)
    Both should be array-like, either 1D or 2D.
     If given 1D arrays, will plot a single spectrum.
     If given 2D arrays, first dimension should correspond to spectrum index.
    label: optional string to include as label
    multi: if True, expects first index of entries to be index, and places
           all plots together on a axis.
    label_coord: the x-coordinate at which to place the label
    fig: the figure to add this plot to
    binning: if set, bins the input arrays by that factor before plotting
    '''
    if binning != None:
        if multi:
            for i,lll in enumerate(lam):
                lam[i] = rebin1D( lll, binning )
            for i,fff in enumerate(flam):
                flam[i] = rebin1D( fff, binning )
            if err != None:
                for i,eee in enumerate(err):
                    err[i] = rebin1D( eee, binning )
        else:
            lam = rebin1D( lam, binning )
            flam = rebin1D( flam, binning )
            if err != None:
                err = rebin1D( err, binning )
        drawstyle='steps-mid'
    else:
        drawstyle='default'
        
    spec_kwargs = dict( alpha=1., linewidth=1, c=(30./256, 60./256, 75./256), drawstyle=drawstyle )
    err_kwargs = dict( interpolate=True, color=(0./256, 165./256, 256./256), alpha=.1 )
    if fig == None:
        fig = plt.figure( figsize=(14,7) )
    ax = plt.subplot(1,1,1)
    
    if not multi:
        ax.plot( lam, flam, **spec_kwargs )
        if err != None:
            ax.fill_between( lam, flam+err, flam-err, **err_kwargs )
        if label != None:
            # put the label where requested, or at a reasonable spot if not requested
            if label_coord == None:
                lam_c = np.max(lam) - (np.max(lam)-np.mean(lam))/3.
            else:
                lam_c = label_coord
            i_c = np.argmin( np.abs(lam-lam_c) )
            flam_c = np.max( flam[i_c:i_c+100] )
            ax.annotate( label, (lam_c, flam_c) )
    else:
        # use data ranges to define an offset for each spectrum
        rngs = [ max(ff)-min(ff) for ff in flam ]
        offset = 0
        for iii in range( len(lam) ):
            if iii != 0:
                offset += rngs[iii-1]
            l, f = lam[iii], flam[iii]+offset
            ax.plot( l, f, **spec_kwargs )
            if err != None:
                e = err[iii]
                if e != None:
                    ax.fill_between( l, f+e, f-e, **err_kwargs )
            if label != None:
                # put the label where requested, or at a reasonable spot if not requested
                if label_coord == None:
                    lam_c = np.max(l) - (np.max(l)-np.mean(l))/3.
                else:
                    lam_c = label_coord[iii]
                i_c = np.argmin( np.abs(l-lam_c) )
                flam_c = np.max( f[i_c:i_c+100] )
                ax.annotate( label[iii], (lam_c, flam_c) )
    
    plt.xlabel(r'Wavelength ($\AA$)')
    plt.ylabel(r'Flux (erg sec$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    return fig


def parse_splot( input ):
    '''
    Parses a string of splot.log output, returning arrays for each column.
    Note: each array may correspond to a different value, depending on which
     splot task was used.
    '''
    out = []
    for line in input.split('\n'):
        vals = [v for v in line.split(' ') if v]
        if not vals: continue
        try:
            out.append( map(float,vals) )
        except:
            # probably a header line; just ignore it
            continue
    return np.array( out )


def identify_matches( queried_stars, found_stars, match_radius=1. ):
    '''
    Use a kd-tree (3d) to match two lists of stars, using full spherical coordinate distances.

    queried_stars, found_stars: numpy arrays of [ [ra,dec],[ra,dec], ... ] (all in decimal degrees)
    match_radius: max distance (in arcseconds) allowed to identify a match between two stars.

    Returns two arrays corresponding to queried stars:
    indices - an array of the indices (in found_stars) of the best match. Invalid (negative) index if no matches found.
    distances - an array of the distances to the closest match. NaN if no match found.
    '''
    ra1, dec1 = queried_stars[:,0], queried_stars[:,1]
    ra2, dec2 = found_stars[:,0], found_stars[:,1]
    dist = 2.778e-4*match_radius # convert arcseconds into degrees 

    cosd = lambda x : np.cos(np.deg2rad(x))
    sind = lambda x : np.sin(np.deg2rad(x))
    mindist = 2 * sind(dist/2) 
    getxyz = lambda r, d: [cosd(r)*cosd(d), sind(r)*cosd(d), sind(d)]
    xyz1 = np.array(getxyz(ra1, dec1))
    xyz2 = np.array(getxyz(ra2, dec2))

    tree2 = scipy.spatial.KDTree(xyz2.transpose())
    ret = tree2.query(xyz1.transpose(), 1, 0, 2, mindist)
    dist, ind = ret
    dist = np.rad2deg(2*np.arcsin(dist/2))

    ind[ np.isnan(dist) ] = -9999
    return ind, dist


def rebin1D( a, factor ):
    '''
    Rebin a 1D array into another array *factor* times smaller.
    
    a: array-like
    factor: integer
    
    If plotting, use keyword argument: drawstyle='steps-mid'
     to represent binning accurately.
    '''
    a = np.array(a)
    assert len(a.shape) == 1
    len_orig = a.shape[0]
    # find the part of the array that is factorable using integer division
    factorable = a[:(len_orig/factor)*factor]
    unfactorable = a[(len_orig/factor)*factor:]
    # perform the rebinning on the factorable part
    arr1 = factorable.reshape( len_orig/factor, factor ).mean(1)
    # and take the average of the unfactorable part
    arr2 = np.array( np.mean(unfactorable) )
    
    return np.hstack( (arr1, arr2) )

def smooth( x, y, width=None, window='hanning' ):
    '''
    Smooth the input spectrum y (on wl x) with a <window> kernel
     of width ~ width (in x units)
    If width is not given, chooses an optimal smoothing width.
    <window> options: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    Returns the smoothed y array.
    '''
    if width == None:
        ys,l = datasmooth.smooth_data(x,y,midpointrule=True)
        print 'chose smoothing lambda of',l
        return ys
    # if given an explicit width, do it all out here
    if y.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size != y.size:
        raise ValueError, "Input x,y vectors must be of same size"
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    avg_width = np.abs(np.mean(x[1:]-x[:-1]))
    window_len = int(round(width/avg_width))
    if y.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return y

    s=np.r_[y[window_len-1:0:-1],y,y[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    yout = y[(window_len/2):-(window_len/2)]
    if len(yout) != len(x):
        yout = yout[:len(x)]
    return yout

def estimate_noise( x, y, plot=False, factor=20 ):
    '''
    Estimate the noise (and therefore error) of the spectrum as 
     a function of wavelength by subtracting a smoothed spectrum.
    '''
    res = np.mean( x[1:] - x[:-1] )
    model = smooth( x, y, factor*res )
    err = generic_filter( y-model, np.std, int(factor*res) )
    if plot:
        plt.figure()
        plt.plot(x,y,'k')
        plt.plot(x,model,'b')
        plt.plot(x,err,'r')
        plt.title('Data, smoothed model, and error estimate.')
        plt.show()
    return err


def rolling_window(a, window):
    # quick way to produce rolling windows of <window> size in numpy
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    middle = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    # now fill in the rest, reflecting about the ends, to make it the same shape as input
    out = np.empty( (a.shape[-1], shape[-1]) )
    begin = int( window/2 )
    end   = int( window-1-window/2 )
    out[:begin] = middle[0]
    out[begin:-end] = middle
    out[end:] = middle[-1]
    return out

def rolling_std(a, window):
    # returns a rolling-window STD
    return np.std( rolling_window(a,window), axis=1 )




def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
def line(x, a, b):
    return a+b*x
def gpl(x, a,b, A,mu,sigma):
    return gauss(x,A,mu,sigma)+line(x,a,b)
def ngauss( *params ):
    '''
    Returns an array of the sum of n gaussian curves.
    params must be: [x, n] + [A,mu,sigma]*n
      (i.e. len(params) = 2 + 3*n)
    '''
    x = np.array(params[0])
    n = int(params[1])
    out = np.zeros_like(x)
    for i in 3*np.arange(n):
        out += gauss(x, params[2+i], params[2+i+1], params[2+i+2])
    return out
def ngpl( *params ):
    '''
    Returns an array of the sum of n gaussian curves plus a line.
    params must be: [x, n] + [a,b] + [A,mu,sigma]*n
      (i.e. len(params) = 4 + 3*n)
    '''
    x = np.array(params[0])
    n = int(params[1])
    a,b = params[2:4]
    out = np.zeros_like(x)
    out += line(x, a,b)
    for i in 3*np.arange(n):
        out += gauss(x, params[4+i], params[4+i+1], params[4+i+2])
    return out
def fit_gaussian( x, y, interactive=False, plot=True, floor=True, p0={} ):
    '''
    Fit a straight line plus a single 1D Gaussian profile to the array y on x.
    Returns a dictionary of the best-fit parameters and the numerical array of the
     best fit.
     
    Options:
     interactive=[False,True]
     If True, will ask for graphical input when fitting line.
     If False, will make reasonable assumptions and try to fit the
      line without human input.
      
     plot=[True,False]
     Only relevant if interactive=False.
     If True, will display final fit plot to verify the quality of fit.
     If False, does not display any plots.
     
     floor=[True,False]
     Include a linear noise floor in the fit.
     
     p0: a dictionary including any of {A,mu,sigma}, to force
         inital parameter guesses if desired.  Only used if interactive=False.
    '''
    
    x = np.array(x)
    y = np.array(y)

    if interactive:
        # get range from plot
        plt.ion()
        plt.figure( figsize=(12,6) )
        plt.clf()
        plt.plot( x, y )
        plt.title('Click twice to define the x-limits of the feature')
        plt.draw()
        print "Click twice to define the x-limits of the feature"
        [x1,y1],[x2,y2] = plt.ginput(n=2)
        # redraw to only that range
        xmin, xmax = min([x1,x2]), max([x1,x2])
        mask = (xmin<x)&(x<xmax)
        plt.clf()
        plt.plot( x[mask], y[mask] )
        plt.title('Click on the peak, and then at one of the edges of the base')
        sized_ax = plt.axis()
        plt.draw()
        print "Click on the peak, and then at one of the edges of the base"
        [x1,y1],[x2,y2] = plt.ginput(n=2)

        A0 = y1-y2
        mu0 = x1
        sig0 = np.abs( x2-x1 )

        if floor:
            # estimate line parameters
            a0 = np.percentile(y[mask], 5.)
            b0 = 0.
            [a,b, A,mu,sigma], pcov = curve_fit(gpl, x[mask], y[mask], p0=[a0,b0, A0,mu0,sig0])
        else:
            [A,mu,sigma], pcov = curve_fit(gauss, x[mask], y[mask], p0=[A0,mu0,sig0])

        # finally, plot the result
        xplot = np.linspace(min(x[mask]), max(x[mask]), len(x[mask])*100)
        plt.ioff()
        plt.close()
        plt.scatter( x,y, marker='x' )
        if floor:
            plt.plot( xplot, gpl(xplot, a,b, A,mu,sigma), lw=2, c='r' )
        else:
            plt.plot( xplot, gauss(xplot, A,mu,sigma), lw=2, c='r' )
        # plt.title('center: {} -- sigma: {} -- FWHM: {}'.format(round(mu,4), round(sigma,4), round(2.35482*sigma,4)))
        plt.axis( sized_ax )
        plt.show()
        
    else:
        # estimate Gaussian parameters
        A0 = np.max(y)
        imax = np.argmax(y)
        mu0 = x[imax]
        # estimate sigma as the distance needed to get down to halfway between peak and median value
        median = np.median(y)
        sig0 = 1.
        for i,val in enumerate(y[imax:]):
            if np.abs( (val-median)/(A0-median) ) < .5:
                sig0 = np.abs( x[ imax+i ] - x[imax] )
                break
        # any given parameters trump estimates
        try:
            A0 = p0['A']
        except KeyError:
            pass
        try:
            mu0 = p0['mu']
        except KeyError:
            pass
        try:
            sig0 = p0['sigma']
        except KeyError:
            pass
        if floor:
            # estimate line parameters
            a0 = np.percentile(y, 5.)
            b0 = 0.
            [a,b, A,mu,sigma], pcov = curve_fit(gpl, x, y, p0=[a0,b0, A0,mu0,sig0])
        else:
            [A,mu,sigma], pcov = curve_fit(gauss, x, y, p0=[A0,mu0,sig0])

        if plot:
            xplot = np.linspace(min(x), max(x), len(x)*100)
            plt.scatter( x,y, marker='x' )
            if floor:
                plt.plot( xplot, gpl(xplot, a,b, A,mu,sigma), lw=2, c='r' )
            else:
                plt.plot( xplot, gauss(xplot, A,mu,sigma), lw=2, c='r' )
            plt.title('center: {} -- sigma: {} -- FWHM: {}'.format(round(mu,4), round(sigma,4), round(2.35482*sigma,4)))
            plt.show()

    outdict = {'A':A, 'mu':mu, 'sigma':sigma, 'FWHM':2.35482*sigma}
    if floor:
        outdict['line_intercept'] = a
        outdict['line_slope'] = b
        return outdict, gpl(x, a,b, A,mu,sigma)
    else:
        return outdict, gauss(x, A,mu,sigma)


def fit_n_gaussians( x, y, n, floor=True ):
    '''
    Fits n gaussians to the input vector y on x.
    Requires interaction to determine starting search parameters.
    '''
    x = np.array(x)
    y = np.array(y)
    # get range to fit to
    plt.ion()
    plt.figure( figsize=(12,6) )
    plt.clf()
    plt.plot( x, y )
    plt.title('Click twice to define the x-limits for fitting')
    plt.draw()
    print "Click twice to define the x-limits for fitting"
    [x1,y1],[x2,y2] = plt.ginput(n=2)
    # redraw to only that range
    xmin, xmax = min([x1,x2]), max([x1,x2])
    mask = (xmin<x)&(x<xmax)
    plt.clf()
    plt.plot( x[mask], y[mask] )
    # go through and get initial parameters for each peak
    init_params = []
    sized_ax = plt.axis()
    for i in range(n):
        plt.title('Click the peak of feature {}, and then at one of the edges of the base'.format(i))
        plt.draw()
        print "Click on the peak of feature {}, and then at one of the edges of the base".format(i)
        [x1,y1],[x2,y2] = plt.ginput(n=2)
        A0 = y1
        mu0 = x1
        sig0 = np.abs( x2-x1 )
        init_params += [A0,mu0,sig0]
    if floor:
        # estimate line parameters
        a0 = np.percentile(y[mask], 5.)
        b0 = 0.
        fit_params, pcov = curve_fit(ngpl, x[mask], y[mask], p0=[n,a0,b0]+init_params)
    else:
        fit_params, pcov = curve_fit(ngauss, x[mask], y[mask], p0=[n]+init_params)

    # plot the result
    xplot = np.linspace(min(x[mask]), max(x[mask]), len(x[mask])*100)
    plt.ioff()
    plt.close()
    plt.scatter( x,y, marker='x' )
    params = [xplot] + fit_params.tolist()
    if floor:
        plt.plot( xplot, ngpl(*params), lw=2, c='r' )
    else:
        plt.plot( xplot, ngauss(*params), lw=2, c='r' )
    plt.title('Best fit with {} Gaussians'.format(n))
    plt.axis( sized_ax )
    plt.show()

    if floor:
        outdict = { 'line':{'a':fit_params[1], 'b':fit_params[2]} }
        for i in range(n):
            A,mu,sig = fit_params[3+3*i : 3+3*i+3]
            outdict['G{}'.format(i)] = {'A':A,'mu':mu,'sigma':sig}
    else:
        outdict = {}
        for i in range(n):
            A,mu,sig = fit_params[1+3*i : 1+3*i+3]
            outdict['G{}'.format(i)] = {'A':A,'mu':mu,'sigma':sig}
    return outdict    

def bb_fit( x, T, Factor ):
    '''used in fit_blackbody, this function returns a blackbody of
       temperature T, evaluated at wavelengths x (Angstroms), and
       rescaled by a factor Factor
    '''
    return Factor * black_body_lam( x, T )


def fit_blackbody( lam, flux, interactive=True, guessT=10000, plot=True):
    '''Fit a blackbody curve to an input spectrum.
       lam: wavelength (A)
       flux: flux (flam)
    '''
    x = np.array(lam)
    y = np.array(flux)

    if interactive:
        plt.ion()
        plt.figure()
        plt.show()
        plt.clf()
        plt.plot( x, y, alpha=.75 )
        plt.title('Click to define points')
        plt.draw()
        print "Click on points to which you'd like to fit the BB curve."
        print " Left click: choose a point"
        print " Right click: remove most recent point"
        print " Middle click: done"
        points = plt.ginput(n=0, timeout=0)
        xPoints, yPoints = map( np.array, zip(*points) )

        # pick a first-guess scale factor
        guessA = yPoints[0]/black_body_lam( xPoints[0], guessT )

        [T,A], pcov = curve_fit(bb_fit, xPoints, yPoints, p0=[guessT, guessA], maxfev=100000)

        print 'Best-fit T:', round(T, 4)

        plt.plot( x, bb_fit(x, T, A), 'k:', lw=2 )
        plt.title( 'best-fit result - click to dismiss' )
        plt.draw()
        print ' (click plot to dismiss)'
        tmp = plt.ginput(n=1, timeout=120)
        plt.ioff()
        plt.close()

    else:
        # use an average value to get a best-guess scale factor
        l = len(x)
        xavg = np.mean(x[int(.25*l):int(.25*l)+10])
        yavg = np.mean(y[int(.25*l):int(.25*l)+10])
        guessA = yavg/black_body_lam( xavg, guessT )

        [T,A], pcov = curve_fit(bb_fit, x, y, p0=[guessT, guessA], maxfev=100000)

        if plot:
            plt.plot( x, y, alpha=.75 )
            plt.plot( x, bb_fit(x, T, A), 'k:', lw=2 )
            plt.title( 'best-fit result' )
            plt.show()

    return T, A


def integrate_line_flux( wl, fl, n=1 ):
    '''
    An interactive script to measure the flux in a line by subtracting a 
    linear continuum fit to regions chosen to either side of the line
    and then integrating the flux.
    '''
    wl = np.array( wl )
    fl = np.array( fl )
    measurements = []
    plt.figure( figsize=(12,6) )
    plt.ion()
    for trial in range(n):
        if n>1: print trial, 'out of', n
        plt.clf()
        plt.plot( wl, fl )
        if trial == 0:
            # choose a windowing range
            print 'Click twice to define the region of general interest'
            plt.title('Click twice to define the corners of the box of general interest')
            plt.draw()
            [x1,y1], [x2,y2] = plt.ginput(2)
            axsize = [min([x1,x2]), max([x1,x2]), min([y1,y2]), max([y1,y2])]
        plt.axis( axsize )
        plt.draw()
        
        print 'Click on the edges of the range to which you would like to fit the left side of the continuum'
        print ' For example:'
        print ' 5750 -> 5800\n'
        plt.title('choose left range')
        [x1,y1],[x2,y1] = plt.ginput(n=2)
        left_cont = [x1,x2]
        print 'Repeat for the right side of the continuum'
        print ' For example:'
        print ' 6000 -> 6200\n'
        plt.title('choose right range')
        plt.draw()
        [x1,y1],[x2,y1] = plt.ginput(n=2)
        right_cont = [x1,x2]

        x1 = np.mean( left_cont )
        x2 = np.mean( right_cont )
        y1 = np.mean( fl[ (min(left_cont)<wl) & (wl<max(left_cont)) ])
        y2 = np.mean( fl[ (min(right_cont)<wl) & (wl<max(right_cont)) ])
        dydx = (y2-y1)/(x2-x1)
        a = y1 - dydx*x1

        # now actually perform the subtraction
        x = wl[ (x1<wl) & (wl<x2) ]
        y = fl[ (x1<wl) & (wl<x2) ] - (a + dydx*x)

        # and the integration
        print 'total flux in the line is'
        print ' ', np.sum(y)
        measurements.append( np.sum(y) )
        
        plt.clf()
        plt.plot( x, y )
        plt.title( 'inspect result. click once to continue.' )
        plt.draw()
        tmp = plt.ginput(n=1)
    plt.ioff()
    plt.close()
    if n>1:
        return measurements
    else:
        return measurements[0]


def fit_and_remove_continuum(X,Y, percentile=95, smoothness=1E8, bins=5,
                                  straightness=1E-4, maxiter=10, plot=0):
    ''' continuum-fitting algorithm for spectra.
    Iteratively follows these steps:
      run a moving-percentile filter to find most likely upper threshold
       >> adust [binsize] and [percentile] to improve this first pass
      LS-fit a spline to the filtered spectrum
       >> adjust [smoothness] to increase/decrease final smoothness
      finally, subtract that spline from input and return both the
       rectified spectrum and the continuum
    Returns:
     (cleaned Y, continuum)
     
    bins: number of bins to use when smoothing.  Many bins: less smooth, few bins: more smooth.
    smoothness: the smoothness parameter for the spline fit; high value means very smooth (few knots)
    straightness: adjust the required flattening. This script stops iterating
      when the sum of the subtracted continuum is a factor of <<smoothness>>
      smaller than the sum of the input vector.
    maxiter: maximum number of iterations, regardless of straightness parameter.
    plot: if True, will show a plot of the outcome
    '''
    flat = np.array(Y)
    cont = np.zeros_like(Y)
    base_val = np.sum(Y)
    binsize = int(len(Y)/bins)
    for i in range(maxiter):
        y1 = percentile_filter(flat, percentile, size=binsize)
        spline = UnivariateSpline( X,y1, s=smoothness )
        y2 = spline(X)
        cont += y2
        flat = Y-cont
        curr_val = np.sum(y2)
        if np.abs(curr_val/base_val) < straightness:
            break
    if plot:
        plt.plot(X,Y,'k', X,cont,'r', X,flat,'g')
        plt.xlabel('wavelength')
        plt.ylabel('flux')
        plt.title('spectrum and continuum fit')
        plt.show()
    return flat, cont


def parse_ra( inn ):
    '''
    Parse input RA string, either decimal degrees or sexagesimal HH:MM:SS.SS (or similar variants).
    Returns decimal degrees.
    '''
    # if simple float, assume decimal degrees
    try:
        ra = float(inn)
        return ra
    except:
        # try to parse with phmsdms:
        res = parse_sexagesimal(inn)
        ra = 15.*( res['vals'][0] + res['vals'][1]/60. + res['vals'][2]/3600. )
        return ra


def parse_dec( inn ):
    '''
    Parse input Dec string, either decimal degrees or sexagesimal DD:MM:SS.SS (or similar variants).
    Returns decimal degrees.
    '''
    # if simple float, assume decimal degrees
    try:
        dec = float(inn)
        return dec
    except:
        # try to parse with phmsdms:
        res = parse_sexagesimal(inn)
        dec = res['sign']*( res['vals'][0] + res['vals'][1]/60. + res['vals'][2]/3600. )
        return dec


def parse_sexagesimal(hmsdms):
    """
    +++ Pulled from python package 'angles' +++
    Parse a string containing a sexagesimal number.
    
    This can handle several types of delimiters and will process
    reasonably valid strings. See examples.
    
    Parameters
    ----------
    hmsdms : str
        String containing a sexagesimal number.
    
    Returns
    -------
    d : dict
    
        parts : a 3 element list of floats
            The three parts of the sexagesimal number that were
            identified.
        vals : 3 element list of floats
            The numerical values of the three parts of the sexagesimal
            number.
        sign : int
            Sign of the sexagesimal number; 1 for positive and -1 for
            negative.
        units : {"degrees", "hours"}
            The units of the sexagesimal number. This is infered from
            the characters present in the string. If it a pure number
            then units is "degrees".
    """
    units = None
    sign = None
    # Floating point regex:
    # http://www.regular-expressions.info/floatingpoint.html
    #
    # pattern1: find a decimal number (int or float) and any
    # characters following it upto the next decimal number.  [^0-9\-+]*
    # => keep gathering elements until we get to a digit, a - or a
    # +. These three indicates the possible start of the next number.
    pattern1 = re.compile(r"([-+]?[0-9]*\.?[0-9]+[^0-9\-+]*)")
    # pattern2: find decimal number (int or float) in string.
    pattern2 = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")
    hmsdms = hmsdms.lower()
    hdlist = pattern1.findall(hmsdms)
    parts = [None, None, None]
    
    def _fill_right_not_none():
        # Find the pos. where parts is not None. Next value must
        # be inserted to the right of this. If this is 2 then we have
        # already filled seconds part, raise exception. If this is 1
        # then fill 2. If this is 0 fill 1. If none of these then fill
        # 0.
        rp = reversed(parts)
        for i, j in enumerate(rp):
            if j is not None:
                break
        if  i == 0:
            # Seconds part already filled.
            raise ValueError("Invalid string.")
        elif i == 1:
            parts[2] = v
        elif i == 2:
            # Either parts[0] is None so fill it, or it is filled
            # and hence fill parts[1].
            if parts[0] is None:
                parts[0] = v
            else:
                parts[1] = v
    
    for valun in hdlist:
        try:
            # See if this is pure number.
            v = float(valun)
            # Sexagesimal part cannot be determined. So guess it by
            # seeing which all parts have already been identified.
            _fill_right_not_none()
        except ValueError:
            # Not a pure number. Infer sexagesimal part from the
            # suffix.
            if "hh" in valun or "h" in valun:
                m = pattern2.search(valun)
                parts[0] = float(valun[m.start():m.end()])
                units = "hours"
            if "dd" in valun or "d" in valun:
                m = pattern2.search(valun)
                parts[0] = float(valun[m.start():m.end()])
                units = "degrees"
            if "mm" in valun or "m" in valun:
                m = pattern2.search(valun)
                parts[1] = float(valun[m.start():m.end()])
            if "ss" in valun or "s" in valun:
                m = pattern2.search(valun)
                parts[2] = float(valun[m.start():m.end()])
            if "'" in valun:
                m = pattern2.search(valun)
                parts[1] = float(valun[m.start():m.end()])
            if '"' in valun:
                m = pattern2.search(valun)
                parts[2] = float(valun[m.start():m.end()])
            if ":" in valun:
                # Sexagesimal part cannot be determined. So guess it by
                # seeing which all parts have already been identified.
                v = valun.replace(":", "")
                v = float(v)
                _fill_right_not_none()
        if not units:
            units = "degrees"
    
    # Find sign. Only the first identified part can have a -ve sign.
    for i in parts:
        if i and i < 0.0:
            if sign is None:
                sign = -1
            else:
                raise ValueError("Only one number can be negative.")
    
    if sign is None:  # None of these are negative.
        sign = 1
    
    vals = [abs(i) if i is not None else 0.0 for i in parts]
    return dict(sign=sign, units=units, vals=vals, parts=parts)


def date2jd( d ):
    return sum(gcal2jd(d.year,d.month,d.day)) + d.hour/24. + d.minute/3600. + d.second/86400.


def doppcor( x, val, velocity=True ):
    '''
    Doppler correct a 1D wavelength array x (Angstroms) by val
       (if velocity=True, val is km/s, if velocity=False, val is z)
    Returns a corrected x array of same shape.
    '''
    x = np.array(x)
    if velocity:
        newx = x - x*(val/2.998E5)
    else:
        newx = x/(1.0+val)
    return newx


def parse_nist_table( s ):
    wls, aks = [],[]
    for row in s.split('\n'):
        try:
            vals = row.split('|')
            wl = float(vals[0])
            ak = float(vals[1])
        except:
            continue
        wls.append(wl)
        aks.append(ak)
    return np.array(wls), np.array(aks)

def query_nist(ion, low_wl, upp_wl, n_out=20):
    '''
    An interface to the NIST website. This function performs
     a query for atomic lines of a certain ion.
    Grabs the <n_out> strongest lines of <ion> in the range
     [<low_wl>, <upp_wl>].
    Returns: wls, strengths (Ak, in units of 10^8 s^-1)
    '''

    br = mechanize.Browser()
    ws = 'http://physics.nist.gov/PhysRefData/ASD/lines_form.html'
    br.open(ws)
    f = br.forms().next() # 'f' is our primary interface to the site

    f.set_value(ion, name='spectra')
    f.set_value(str(low_wl), name='low_wl')    # set the upper and lower wavelengths (in A)
    f.set_value(str(upp_wl), name='upp_wl')

    # select Angstroms for our WL units and an ascii table for output (no javascript)
    c = f.find_control(name='unit')
    c.items[0].selected = True
    c = f.find_control(name='format')
    c.items[1].selected = True
    c = f.find_control(name='remove_js')
    c.items[0].selected = True

    # only return lines with observed wavelengths
    c = f.find_control(name='line_out')
    c.items[3].selected = True

    # return Ak in 10^8 s-1 units
    c = f.find_control(name='A8')
    c.items[0].selected = True

    # deselect a host of irrelevant outputs
    to_deselect = ['bibrefs', 'show_calc_wl', 'intens_out', 'conf_out', 'term_out', 'enrg_out', 'J_out']
    for n in to_deselect:
        c = f.find_control(name=n)
        c.items[0].selected = False

    response = mechanize.urlopen(f.click())
    html = response.read()
    tab = html.split('<pre>')[1].split('</pre>')[0]
    wls, aks = parse_nist_table( tab )

    # return the strongest lines
    mask = np.argsort(aks)[:-n_out:-1]
    return wls[mask], aks[mask]

class lookatme:
    '''
    A tool to look at the properties of a SN spectrum.
    '''

    def __init__(self, wl, fl, er=None, deredden=None, dez=None):
        '''
        wl,fl: 1D wavelength and flux vectors (wl in Angstroms)
        er: optional 1D array of spectral errors
        deredden: if given a float, will assume it is E(B-V) and deredden the spectrum
                  if given a tuple of (ra,dec), will deredden by the MW absorption along that line of sight
        dez: given a float value for z, will deredshift the spectrum
        '''
    
        self.wl = np.array(wl)
        self.orig_wl = np.array(wl)
        self.fl = np.array(fl)
        self.orig_fl = np.array(fl)
        if er!=None:
            self.er = np.array(er)
        else:
            self.er = None
        # deredden, if desired
        if deredden:
            print 'dereddening spectrum'
            if type(deredden) == tuple:
                ra,dec = deredden
                self.fl, self.EBV = dered.remove_galactic_reddening( ra, dec, self.wl, self.fl, verbose=True )
            else:
                self.fl = dered.dered_CCM( wl, fl, deredden )
                self.EBV = deredden
        else:
            self.EBV = 0.0
        if dez:
            print 'deredshifting spectrum'
            self.wl = doppcor( self.wl, dez, velocity=False )
            self.dopcor = dez
        else:
            self.dopcor = 0.0
        self.ion_lines = {}
        self.fitted_lines = []
        self.ion_dict = {100:'H I', 200:'He I', 201:'He II', 300:'Li I', 301:'Li II', 400:'Be I',
                401:'Be II', 402:'Be III', 500:'B I', 501:'B II', 502:'B III', 503:'B IV',
                600:'C I', 601:'C II', 602:'C III', 603:'C IV', 700:'N I', 701:'N II',
                702:'N III', 703:'N IV', 704:'N V', 800:'O I', 801:'O II', 802:'O III',
                803:'O IV', 804:'O V', 900:'F I', 901:'F II', 1000:'Ne I', 1100:'Na I',
                1200:'Mg I', 1201:'Mg II', 1300:'Al I', 1301:'Al II', 1302:'Al III', 1400:'Si I',
                1401:'Si II', 1402:'Si III', 1403:'Si IV', 1500:'P I', 1501:'P II', 1502:'P III',
                1600:'S I', 1601:'S II', 1602:'S III', 1700:'Cl I', 1800:'Ar I', 1801:'Ar II',
                1900:'K I', 1901:'K II', 2000:'Ca I', 2001:'Ca II', 2100:'Sc I', 2101:'Sc II',
                2200:'Ti I', 2201:'Ti II', 2202:'Ti III', 2300:'V I', 2301:'V II', 2302:'V III',
                2400:'Cr I', 2401:'Cr II', 2402:'Cr III', 2500:'Mn I', 2501:'Mn II', 2502:'Mn III',
                2600:'Fe I', 2601:'Fe II', 2602:'Fe III', 2603:'Fe IV', 2700:'Co I', 2701:'Co II',
                2702:'Co III', 2703:'Co IV', 2800:'Ni I', 2801:'Ni II', 2802:'Ni III', 2803:'Ni IV',
                2901:'Cu II', 3002:'Zn III', 3801:'Sr II', 5600:'Ba I', 5601:'Ba II'}
        self.simple_lines = {'H I': [3970, 4102, 4341, 4861, 6563], 'Na I': [5890, 5896, 8183, 8195],
                'He I': [3886, 4472, 5876, 6678, 7065], 'Mg I': [2780, 2852, 3829, 3832, 3838, 4571, 5167, 5173, 5184],
                'He II': [3203, 4686], 'Mg II': [2791, 2796, 2803, 4481], 'O I': [7772, 7774, 7775, 8447, 9266],
                'Si II': [3856, 5041, 5056, 5670, 6347, 6371], 'O II': [3727],
                'Ca II': [3934, 3969, 7292, 7324, 8498, 8542, 8662], 'O III': [4959, 5007],
                'Fe II': [5018, 5169, 7155], 'S II': [5433, 5454, 5606, 5640, 5647, 6715],
                'Fe III': [4397, 4421, 4432, 5129, 5158], '[O I]':[6300, 6364],
                '[Ca II]':[7291,7324], 'Mg I]':[4571] }
        # use a subset of mpl colors 
        self.allcolors = [c for c in colors.cnames.keys() if ('dark' in c) or ('medium') in c ] +\
                          'r g b c m k'.split()
        plt.ion()
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er)
        self.leg = None
        plt.show()
        plt.draw()
        self.run()

    def reset(self):
        self.fig.clf()
        self.wl = self.orig_wl
        self.fl = self.orig_fl
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er, fig=self.fig)
        self.leg = None
        print 'clearing and redrawing spectrum'
        plt.draw()

    def add_nist_lines(self, ion, v=0.0):
        try:
            print 'querying NIST for',ion,'lines'
            wls, aks = query_nist(ion, np.min(self.wl), np.max(self.wl))
        except:
            print 'query failed! no internet? maybe a bad ion name?'
            raise Exception
        ymin,ymax = plt.axis()[2:]
        color = self.allcolors[ np.random.randint(len(self.allcolors)) ]
        print 'adding',color,'lines for',ion,'at velocity',v
        wls = wls - wls*(v/3e5)
        self.ion_lines[ion] = plt.vlines( wls, ymin, ymax, linestyles='dashed', lw=2, color=color, label=ion )
        self.leg = plt.legend(fancybox=True, loc='best' )
        self.leg.get_frame().set_alpha(0.0)
        plt.draw()


    def add_simple_lines(self, ion, v=0.0):
        try:
            wls = np.array(self.simple_lines[ion])
            ymin,ymax = plt.axis()[2:]
            color = self.allcolors[ np.random.randint(len(self.allcolors)) ]
            print 'adding',color,'lines for',ion,'at velocity',v
            wls = wls - wls*(v/3e5)
            self.ion_lines[ion] = plt.vlines( wls, ymin, ymax, linestyles='dashed', lw=2, color=color, label=ion )
            self.leg = plt.legend(fancybox=True, loc='best' )
            self.leg.get_frame().set_alpha(0.0)
            plt.draw()
        except:
            print 'query failed! maybe a bad ion name?'

    def remove_lines(self, ions):
        print 'removing lines for',ions
        if type(ions) == list:
            # ions is a list of ions
            for ion in ions:
                self.ion_lines[ion].remove()
                self.ion_lines.pop(ion)
        else:
            # ions is a single ion
            self.ion_lines[ions].remove()
            self.ion_lines.pop(ions)
        self.leg = plt.legend(fancybox=True, loc='best' )
        if self.leg:
            self.leg.get_frame().set_alpha(0.0)
        plt.draw()

    def modify_dopcor(self, newz):
        if self.dopcor:
            print 'removing previous redshift correction:',self.dopcor
            self.wl = self.orig_wl
        print 'correcting for a redshift of',newz
        self.wl = doppcor( self.wl, newz, velocity=False )
        self.dopcor = newz
        plt.clf()
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er, fig=self.fig)
        for ion in self.ion_lines.keys():
            self.ion_lines.pop(ion)
        plt.draw()
    
    def deredden(self, EBV):
        print 'dereddenening: E(B-V) =',EBV
        self.fl = dered.dered_CCM( self.wl, self.fl, EBV )
        self.EBV += EBV
        plt.clf()
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er, fig=self.fig)
        for ion in self.ion_lines.keys():
            self.ion_lines.pop(ion)
        plt.draw()
    
    def deredden_gal(self, srchstr):
        print 'dereddening using MW values'
        self.fl, EBV = dered.remove_galactic_reddening( srchstr, self.wl, self.fl, verbose=True )
        print 'using E(B-V) = {}'.format(EBV)
        plt.clf()
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er, fig=self.fig)
        for ion in self.ion_lines.keys():
            self.ion_lines.pop(ion)
        plt.draw()

    def smooth(self, width):
        print 'smoothing. width:',width
        self.fl = smooth(self.wl, self.fl, width)
        plt.clf()
        self.fig = pretty_plot_spectra(self.wl, self.fl, err=self.er, fig=self.fig)
        for ion in self.ion_lines.keys():
            self.ion_lines.pop(ion)
        plt.draw()

    def fit_abs_line(self, xmid):
        print 'fitting an absorption line around',xmid
        xx, yy, ee, pc, yy2 = spectools.parameterize_line(self.wl, self.fl, self.er, xmid)
        self.fitted_lines.append(plt.plot(xx,yy2,'r')[0])
        self.fitted_lines.append(plt.plot(xx,pc,'k',lw=2)[0])
        pew,wl_mid,rel_d,fwhm = spectools.calc_everything(self.wl, self.fl, self.er, xmid)
        print ' pEW: %.3f\n wl_mid: %.3f\n rel_depth: %.3f\n FWHM: %.3f\n Integral: %.3f\n' \
              %(pew,wl_mid,rel_d,fwhm, trapz(yy2-pc, x=xx))
        plt.draw()

    def fit_emis_line(self, xmid):
        print 'fitting an emission line around',xmid
        xx, yy, ee, pc, yy2 = spectools.parameterize_line(self.wl, self.fl, self.er, xmid, emission=True)
        self.fitted_lines.append(plt.plot(xx,yy2,'r')[0])
        self.fitted_lines.append(plt.plot(xx,pc,'k',lw=2)[0])
        pew,wl_mid,rel_d,fwhm = spectools.calc_everything(self.wl, self.fl, self.er, xmid, emission=True)
        print ' pEW: %.3f\n wl_mid: %.3f\n rel_height: %.3f\n FWHM: %.3f\n Integral: %.3f\n' \
               %(pew,wl_mid,rel_d,fwhm, trapz(yy2-pc, x=xx))
        plt.draw()

    def fit_gauss_line(self):
        print 'fitting a gaussian'
        print 'click to mark the region of the spectrum to consider'
        [x1,y1], [x2,y2] = plt.ginput(2)
        mask = (self.wl>min([x1,x2])) & (self.wl<max([x1,x2]))
        gparams, garray = fit_gaussian(self.wl[mask], self.fl[mask], interactive=True)
        self.fitted_lines.append(plt.plot(self.wl[mask],garray,'r'))
        # get the Gaussian alone, w/o the line
        garray = gauss( self.wl[mask], gparams['A'], gparams['mu'], gparams['sigma'] )
        print ' A: %.3f\n mu: %.3f\n sigma: %.3f\n FWHM: %.3f\n Integral: %.3f\n' \
              %(gparams['A'], gparams['mu'], gparams['sigma'], gparams['FWHM'], trapz(garray, x=self.wl[mask]))
        plt.draw()

    def remove_fitted_lines(self):
        print 'removing all fitted lines'
        for l in self.fitted_lines:
            if type(l) == list:
                for ll in l:
                    ll.remove()
            else:
                l.remove()
        self.fitted_lines = []
        plt.draw()
    
    def run(self):
        '''
        Wait for input and respond to it.
        '''
        while True:
            inn = raw_input('\nLookatme!\n\nOptions:\n'+\
                            ' l: add/remove ion lines to the plot\n'+\
                            ' z: deredshift the spectrum (removing any previous redshifts)\n'+\
                            ' r: deredden the spectrum (without removing any previous reddening)\n'+\
                            ' s: smooth the spectrum\n'+\
                            ' f: fit for a spectral line\n'+\
                            ' c: execute a python command\n'+\
                            ' u: undo everything and replot spectrum\n'+\
                            ' q: quit\n')
            if 'l' in inn.lower():
                inn = raw_input('\nenter an ion name (e.g. "Fe II") and an optional blueshift velocity,'+\
                                'or "clear" to remove all lines, or "nist" to add lines from the NIST database.\n')
                if 'clear' in inn:
                    self.remove_lines( self.ion_lines.keys() )
                elif 'nist' in inn:
                    inn = raw_input('\nenter an ion name (e.g. "Fe II") and an optional blueshift velocity.\n')
                    ion = re.search('[^0-9]+', inn).group().strip()
                    if ion in self.ion_lines.keys():
                        self.remove_lines(ion)
                    else:
                        try:
                            v = float(re.search('[0-9]+', inn).group())
                        except:
                            v = 0.0
                        self.add_nist_lines(ion, v)
                else:
                    ion = re.search('[^0-9]+', inn).group().strip()
                    if ion in self.ion_lines.keys():
                        self.remove_lines(ion)
                    else:
                        try:
                            v = float(re.search('[0-9]+', inn).group())
                        except:
                            v = 0.0
                        self.add_simple_lines(ion, v)
            elif 'z' in inn.lower():
                inn = raw_input('\nenter a numerical redshift z\n')
                try:
                    newz = float(inn)
                except:
                    print 'You entered:',inn
                    print 'I do not understand.'
                    continue
                self.modify_dopcor(newz)
            elif 'r' in inn.lower():
                inn = raw_input('\nenter the numerical E(B-V) to deredden, or hit enter to use MW values\n')
                try:
                    EBV = float(inn)
                    self.deredden(EBV)
                except:
                    inn = raw_input('\nenter a SIMBAD-resolvable search string (coords or name)\n')
                    try:
                        self.deredden_gal( inn.strip() )
                    except:
                        print 'Error determining galactic reddening'
                        continue
            elif 's' in inn.lower():
                inn = raw_input('\nenter width of the smoothing window (A)\n')
                try:
                    self.smooth(float(inn))
                except Exception as e:
                    print 'You entered:',inn
                    print 'I do not understand:\n',e
                    continue
            elif 'f' in inn.lower():
                inn = raw_input('type "a" to fit an absorption line, "e" for an emission line., '+\
                                ' or "g" to fit a Gaussian interactively. '+\
                                ' then hit enter and click on the line to fit. Or simply type "clear"'+\
                                ' to clear all lines\n')
                if 'clear' in inn.lower():
                    self.remove_fitted_lines()
                elif 'a' in inn.lower():
                    try:
                        x,y = plt.ginput()[0]
                        self.fit_abs_line(x)
                    except Exception as e:
                        print 'something went wrong:',e
                        continue
                elif 'e' in inn.lower():
                    try:
                        x,y = plt.ginput()[0]
                        self.fit_emis_line(x)
                    except Exception as e:
                        print 'something went wrong:',e
                        continue
                elif 'g' in inn.lower():
                    try:
                        self.fit_gauss_line()
                    except Exception as e:
                        print 'something went wrong:',e
                        continue
                else:
                    print 'what?'
                    continue
            elif 'c' in inn.lower():
                inn = raw_input('\nenter the python code to run\n')
                try:
                    exec(inn)
                except Exception as e:
                    print 'You entered:',inn
                    print 'I do not understand:\n',e
                    continue
            elif 'u' in inn.lower():
                self.reset()
                continue
            elif 'q' in inn.lower():
                break
            else:
                print 'You entered:',inn
                print 'I do not understand.'
                continue
                
def open_multispec_fits(f):
    '''
    Opens a fits file created by MULTIFITS, returning
     arrays for wl, flux, error, and background.
    '''
    hdu = pyfits.open(f)[0]
    h = hdu.header
    d = hdu.data
    # make the wl array
    try:
        assert( 'linear' in h['WAT1_001'].lower() )
    except:
        raise ValueError('Wavelength not linear!')
    wl = np.linspace(h['xmin'], h['xmax'], d.shape[-1])
    fl = d[0,0,:]
    er = d[3,0,:]
    bg = d[2,0,:]
    
    return wl,fl,er,bg


