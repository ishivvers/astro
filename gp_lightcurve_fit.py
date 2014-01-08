'''
Use Gaussian Processes to fit the lightcurve of a supernova.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

def gp_lc_fit(t, m, e, npoints=10000, plot=True, theta=0.15):
    '''
    Use Gaussian process regression to fit a functional
     form to a supernova lightcurve.
    t, m, e: 1D arrays for time, mag, and error
    theta: the autcorrelation timescale parameter. If the model needs to have more
            flexture, increase theta.  If it needs less, decrease theta.
    returns: t, mag, err arrays of size npoints
    '''
    edge = 1.0  # predict beyond the data by this amount on either side

    X = np.atleast_2d( t ).T  # need a 2D array for skl procedures
    y = np.array(m)
    e = np.array(e)
    x = np.atleast_2d( np.linspace(t[0]-edge, t[-1]+edge, npoints)).T
    xf = x.flatten()

    gp = GaussianProcess(regr='linear', nugget=(e/y)**2, theta0=theta)
    gp.fit(X,y)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    ep = MSE**.5  # prediction error estimate

    if plot:
        plt.figure()
        plt.errorbar( t, m, yerr=e, fmt='b.' )
        plt.plot(xf, y_pred, 'g', lw=2)
        plt.fill_between( xf, y_pred+ep, y_pred-ep, alpha=0.5, color='g' )
        plt.gca().invert_yaxis()
        plt.title('Photometry, GP model, and errors')
        plt.show()
    return xf, y_pred, ep

def find_phot_peak( x, y, err, plot=False ):
    '''
    Given a smooth lightcurve model y on x with error err,
     find the peak epoch and estimate the error.
    x,y,err: 1d arrays
    returns: peak, error
    '''
    # find peak
    ipeak = np.argmin(y)
    xpeak = x[ipeak]
    # estimate the error in the epoch of peak by measuring
    #  it for the best-fit as well as the two 1sigma extremes,
    #  and using the largest error between them all as an estimate
    #  of the error
    w = 10.0 
    mask = (x > x[ipeak]-w)&(x < x[ipeak]+w)
    lpeak = x[mask][np.argmin( (y-err)[mask] )]
    hpeak = x[mask][np.argmin( (y+err)[mask] )]
    error = np.max(np.abs( [xpeak-lpeak, xpeak-hpeak] ))

    if plot:
        plt.figure()
        plt.plot(x, y, 'g', lw=2)
        plt.fill_between( x, y+err, y-err, alpha=0.5, color='g' )
        ox = plt.axis()
        plt.vlines( [xpeak, lpeak, hpeak], ox[2], ox[3], linestyles='dotted', color='k')
        plt.gca().invert_yaxis()
        plt.title('Lightcurve with identified peaks')
        plt.show()
    return xpeak, error

if __name__ == '__main__':
    import sys
    try:
        d = np.loadtxt(sys.argv[1])
    except:
        raise IOError, "Include a loadtxt-readable data array as first argument!"

    t,m,e = d[:,0], d[:,1], d[:,2]
    xf, y_pred, ep = gp_lc_fit(t,m,e, plot=True)
    xpeak, error = find_phot_peak(xf, y_pred, ep)
    print 'Epoch of max: %.3f +\- %.3f' %(xpeak, error)