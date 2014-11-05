import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev
import sys
import os

class normalizer:
    """
    An interactive spline fitting routine for determining the continuum
     level of a spectrum.

    """

    def __init__(self, wave, flux, window=10.0, name=None, filename=None):
        self.wave = wave
        self.flux = flux
        self.continuum = None
        self.cfunc = None
        fig = plt.figure()
        self.ax = plt.gca()
        # the width of the median window taken at each point
        self.winwidth = window/2.0

        self.ax.plot(self.wave,self.flux,'k-',label='spectrum')
        if name != None:
            plt.title(name)
        self.filename = filename

        # Connect the different functions to the different events
        fig.canvas.mpl_connect('key_press_event',self.ontype)
        fig.canvas.mpl_connect('button_press_event',self.onclick)
        fig.canvas.mpl_connect('pick_event',self.onpick)
        plt.show() # show the window

        print '*'*20
        print ' L click to define spline points'
        print ' R click on a point to remove it'
        print ' press enter at any point to refit continuum'
        print ' after continuum is fit, press "n" to normalize'
        print ' press "r" to reset'
        print ' press "w" to write continuum out to file'
        print 'When finished, the continuum is accessible as "<normalizer>.continuum"'
        print 'The continuum function is accessible as "<normalizer>.cfunc"'
        print '*'*20

    def onclick(self, event):
        # when none of the toolbar buttons is activated and the user clicks in the
        # plot somewhere, compute the median value of the spectrum in a 10angstrom
        # window around the x-coordinate of the clicked point. The y coordinate
        # of the clicked point is not important. Make sure the continuum points
        # `feel` it when it gets clicked, set the `feel-radius` (picker) to 5 points
        toolbar = plt.get_current_fig_manager().toolbar
        if event.button==1 and toolbar.mode=='':
            window = ((event.xdata-self.winwidth)<=self.wave) &\
                      (self.wave<=(event.xdata+self.winwidth))
            y = np.median(self.flux[window])
            self.ax.plot(event.xdata,y,'rs',ms=10,picker=5,label='cont_pnt')
        plt.draw()

    def onpick(self, event):
        # when the user right clicks on a continuum point, remove it
        if event.mouseevent.button==3:
            if hasattr(event.artist,'get_label') and event.artist.get_label()=='cont_pnt':
                event.artist.remove()

    def ontype(self, event):
        # when the user hits enter:
        # 1. Cycle through the artists in the current axes. If it is a continuum
        #    point, remember its coordinates. If it is the fitted continuum from the
        #    previous step, remove it
        # 2. sort the continuum-point-array according to the x-values
        # 3. fit a spline and evaluate it in the wavelength points
        # 4. plot the continuum
        if event.key=='enter':
            cont_pnt_coord = []
            for artist in self.ax.get_children():
                if hasattr(artist,'get_label') and artist.get_label()=='cont_pnt':
                    cont_pnt_coord.append(artist.get_data())
                elif hasattr(artist,'get_label') and artist.get_label()=='continuum':
                    artist.remove()
            cont_pnt_coord = np.array(cont_pnt_coord)[...,0]
            sort_array = np.argsort(cont_pnt_coord[:,0])
            x,y = cont_pnt_coord[sort_array].T
            spline = splrep(x,y,k=3)
            self.continuum = splev(self.wave,spline)
            self.cfunc = lambda w: splev(w, spline)
            plt.plot(self.wave,self.continuum,'r-',lw=2,label='continuum')

        # when the user hits 'n' and a spline-continuum is fitted, normalise the
        # spectrum
        elif event.key=='n':
            if self.continuum is not None:
                self.ax.cla()
                self.ax.plot(self.wave,self.flux/self.continuum,'k-',label='normalised')

        # when the user hits 'r': clear the axes and plot the original spectrum
        elif event.key=='r':
            self.continuum = None
            self.ax.cla()
            self.ax.plot(self.wave,self.flux,'k-')

        # when the user hits 'w': if the normalised spectrum exists, write it to a
        # file.
        elif event.key=='w':
            for artist in self.ax.get_children():
                if hasattr(artist,'get_label') and artist.get_label()=='normalised':
                    data = np.array(artist.get_data())
                    if self.filename == None:
                        self.filename = 'normalised_spec.flm'
                    np.savetxt(os.path.splitext(self.filename)[0]+'.nspec',data.T)
                    print('Saved to file:',self.filename)
                    break
        plt.draw()