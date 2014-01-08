'''
Spectral measurement tools.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def quad(x,a,b,c):
    return a*x**2 + b*x + c

def fit_quad(x,y):
    x = np.array(x)
    y = np.array(y)
    # estimate parameters
    c0 = np.mean(y)
    b0 = a0 = 0.0
    [a,b,c], pcov = curve_fit(quad, x, y, p0=[a0,b0,c0])
    # return best fit and parameters
    return quad(x,a,b,c), [a,b,c]

def find_edge(x, y, xmid, side, width=100.0,  plot=False):
    '''
    Find the edge of a feature by searching for the
     maximum inflection point in a set of locally-fit quadratics.
    x,y: spectrum Y on x
    xmid: an x-coordinate inside the feature
    side: one of 'left','right','l','r'
    width: width (in x-coords) of fitting window
    '''
    if side in ['l','left']:
        side = 'l'
    elif side in ['r','right']:
        side = 'r'
    else:
        raise IOError, "side must be one of 'left','right','l','r'"
    edge = None

    while True:
        mask = (x > xmid-width/2)&(x < xmid+width/2)
        xx = x[ mask ]
        yy = y[ mask ]
        ymod = fit_quad(xx,yy)[0]
        # test to see if we've moved past one of the edges
        if (xx[-1] < x[0]+width/2) or (xx[0] > x[-1]-width/2):
            raise ValueError, "Edge not found."

        imax = np.argmax(ymod)
        if (imax != 0) and (imax != len(xx)-1):
            # we have an edge!
            # use a high percentile of the region inside the feature near the edge
            #  to define our y value for the edge of the feature.
            yval = np.percentile(yy, 90)
            edge = ( xx[imax], yval )
            break
        if side=='l':
            xmid -= width/10.0
        elif side=='r':
            xmid += width/10.0
    if not edge:
        raise ValueError, "Edge not found."
    if plot:
        plt.figure()
        plt.plot(x,y,'b')
        plt.plot(xx, ymod, 'k', lw=2)
        plt.scatter(edge[0], edge[1], marker='D', s=250, c='k', alpha=0.5)
        plt.show()
    return edge

def find_pcont(x, y, err, xmid, plot=False, width=100.0):
    '''
    Find and remove the pseudocontinuum for the line centered
     at xmid (x-coords) in the spectrum y on x.
    <width> is the edge-window width in x-coords
    Returns the line and the pseudocontinuum (x, y, y_pc).
    '''
    # find the edges
    l_edge = find_edge(x,y,xmid,'l', width=width)
    r_edge = find_edge(x,y,xmid,'r', width=width)
    # calculate the line
    b = (r_edge[1]-l_edge[1])/(r_edge[0]-l_edge[0])
    a = l_edge[1] - b*l_edge[0]
    mask = (x>l_edge[0])&(x<r_edge[0])
    xx = x[ mask ]
    yy = y[ mask ]
    if err != None:
        ee = err[ mask ]
    else:
        ee = None
    pc = a + b*xx
    if plot:
        plt.figure()
        plt.plot(x,y,'b')
        plt.plot(xx, pc, 'k', lw=2)
        plt.scatter( [l_edge[0],r_edge[0]], [l_edge[1],r_edge[1]], marker='D', s=150, c='k', alpha=0.5)
        plt.title('pseudo-continuum fit')
        plt.show()
    return xx, yy, ee, pc

def pEW(x, y, pc):
    '''
    Calculates the pseudo-equivalent width as described
     by Silverman '12.
    x: wavelength
    y: flux (on x)
    pc: pseudocontinuum (on x)
    returns the pEW
    '''
    greg = []
    for i in range(len(x)-1):
        dlam = x[i+1] - x[i]
        greg.append( dlam*( (pc[i]-y[i])/pc[i] ) )
    return np.sum(greg)

def FWHM(x, y, peak_val=None, plot=False):
    '''
    Calculates and returns the FWHM for the spectral line y on x.
     Assumes line is a well-behaved absorption line, doesn't work for emission.
     If spectrum is noisy, assumes minimum value is the depth.
     If peak_val is given, will calculate relative to that (instead of y_max)
    '''
    if peak_val == None:
        peak_val = np.max(y)
    depth = peak_val - np.min(y)
    y_hm = peak_val - 0.5 * depth
    # find the intersections with y_hm as you move left to right
    edges = []
    fl, fr = False, False
    for i in range(len(x)):
        # look for left side first
        if not fl:
            if y[i] < y_hm:
                # found the left edge
                fl = True
                # local linear approximation to find xintercept
                b = (y[i]-y[i-1])/(x[i]-x[i-1])
                a = y[i-1] - b*x[i-1]
                xmid = (y_hm-a)/b
                edges.append( xmid )
        else:
            # now start looking for the right side
            if not fr:
                if y[i] > y_hm:
                    # found the right edge
                    fr = True
                    # local linear approximation to find xintercept
                    b = (y[i]-y[i-1])/(x[i]-x[i-1])
                    a = y[i-1] - b*x[i-1]
                    xmid = (y_hm-a)/b
                    edges.append( xmid )
                    break
    try:
        assert len(edges) == 2
    except:
        raise ValueError, "Intersections with HM not found!"
    if plot:
        plt.figure()
        plt.plot(x,y,'b')
        plt.plot(x, y_hm*np.ones_like(x), 'k', lw=2)
        plt.scatter([edges[0],edges[1]], [y_hm, y_hm], marker='D', s=150, c='k', alpha=0.5 )
        plt.title('FWHM measurement')
        plt.show()
    return edges[1] - edges[0]

def parameterize_line(x, y, err, xmid, plot=False, width=100.0, spline_smooth=None, line_container=None,
                      test_fit=False, xyrange=None):
    '''
    Fit a functional form to the line centered at xmid in 
     the spectrum y on x with errors err (all array-like).
    <width> is the edge-window width in x-coords
    <spline_smooth> defines the smoothing parameter for the spline based on the length of the wl array
    <line_container> can be an object to append all plotted lines to, so they can be removed later
    If <test_fit>, this attempts a few sanity checks on the line to discard bad fits, and <xyrange> must
     be a tuple with the xrange and yrange of the full spectrum.
    Returns the line, the pseudocontinuum, the fit form, and the function
     (x, y, y_pc, y_fit, f(x))
    '''
    if line_container != None:
        lc = line_container
    else:
        lc = []
    if spline_smooth == None:
        s = spline_smooth
    else:
        s = spline_smooth * len(x)
    if test_fit and type(xyrange)!=tuple:
        raise Exception('If test_fit is true, must include measures of full spectral range.')
    
    # run sanity checks
    attempts = 0
    max_slope = 3.5      # max slope of pseudocontinuum, relative to width of full spectrum
    max_width = 30000.0  # max width of feature in km/s
    min_width = 5000.0   # min width of feature in km/s
    max_offset = 15000.0 # max central velocity offset of feature in km/s
    while True:
        xx, yy, ee, pc = find_pcont(x, y, err, xmid, width=width)
        spline = UnivariateSpline( xx, yy, k=5, s=s )
        yy2 = spline(xx)
        if attempts > 3:
            raise Exception('Could not successfully fit line!')
        elif test_fit:
            ss = ((max(pc)-min(pc))/xyrange[1]) / ((max(xx)-min(xx))/xyrange[0])
            ww = 3e5 * (max(xx)-min(xx))/xmid
            os = 3e5 * np.abs( (xx[np.argmin(yy2)]) - xmid ) / xmid
            if ss > max_slope:
                print 'slope too steep'
                # if slope is too steep, try making the edge width bigger
                width *= 1.5
                attempts += 1
                continue
            elif ww > max_width:
                print 'feature too wide'
                # if feature is too wide, try making the edge width smaller
                width *= (2./3)
                attempts += 1
                continue
            elif ww < min_width:
                print 'feature too narrow'
                # if feature is too narrow, try making the edge width bigger
                width *= 1.5
                attempts += 1
                continue
            elif os > max_offset:
                print 'feature too far offset from center'
                # probably chose the wrong feature; try making the edge width smaller
                width *= (2./3)
                attempts += 1
                continue
            else:
                break
        else:
            break

    if plot:
        ls = plt.plot(xx,yy,'b')
        for l in ls: lc.append(l)
        ls = plt.plot(xx,yy2,'r')
        for l in ls: lc.append(l)
        ls = plt.plot(xx,pc,'k',lw=2)
        for l in ls: lc.append(l)
        plt.show()
    return xx, yy, ee, pc, yy2

def calc_everything(x, y, err, xmid, plot=0, width=100.0, spline_smooth=None, line_container=None):
    '''
    Calculates properties of a the line centered at xmid in 
     the spectrum y on x.  <plot> can be one of [0,1,2], to produce
     different levels of plots. <width> is the width of the edge windows
     used to fit for the continuum. <spline_smooth> is a factor that defines
     the spline_smooth factor based on size of the wl array.
    Returns:
     pEW
     wl_min
     rel_depth
     FWHM
    '''
    if plot > 0:
        p = True
    else:
        p = False
    # xyrange = (np.max(x)-np.min(x), np.max(y)-np.min(y))
    xyrange = (6000, np.max(y)-np.min(y)) # use a fixed wl range, so that the sanity checks aren't dependant on wl range
    xx, yy, ee, pc, yy2 = parameterize_line(x,y,err,xmid, plot=p, width=width, spline_smooth=spline_smooth,
                                        line_container=line_container, test_fit=True, xyrange=xyrange)
    # the pseudo equivalent width
    pew = pEW(xx,yy,pc)
    # the wl of the minimum
    imin = np.argmin(yy2)
    wl_min = xx[imin]
    # the relative depth of the feature
    relative = yy2/pc
    rel_depth = 1.0 - np.min(relative)
    # the FWHM
    if plot > 1:
        p = True
    else:
        p = False
    fwhm = FWHM(xx, relative, 1.0, plot=p)
    return pew, wl_min, rel_depth, fwhm

