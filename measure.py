'''
Class used to measure the properties of 
 SN spectra.

to do: 
 - feed through parameter changes for PC fitting
'''

import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import re
from astro.spectools import calc_everything
from astro.iAstro import pretty_plot_spectra
import pickle

class measurer():
    '''
    interactively measure the properties of the spectrum
     in file <f>, with lines drawn at <v> km/s blueshift
    '''
    ### Define some class variables ###
    # Dictonaries of relevant lines
    #  Each entry should be a list, even if it's only one element long
    simple_lines = {'H I': [3970, 4102, 4341, 4861, 6563], 'Na I': [5890, 5896, 8183, 8195],
        'He I': [3886, 4472, 5876, 6678, 7065], 'Mg I': [2780, 2852, 3829, 3832, 3838, 4571, 5167, 5173, 5184],
        'He II': [3203, 4686], 'Mg II': [2791, 2796, 2803, 4481], 'O I': [7772, 7774, 7775, 8447, 9266],
        'Si II': [3856, 5041, 5056, 5670, 6347, 6371], 'O II': [3727],
        'Ca II': [3934, 3969, 7292, 7324, 8498, 8542, 8662], 'O III': [4959, 5007],
        'Fe II': [5018, 5169], 'S II': [5433, 5454, 5606, 5640, 5647, 6715],
        'Fe III': [4397, 4421, 4432, 5129, 5158], '[O I]':[6300, 6364] }
    phot_lines = {'H I': [4341, 4861, 6563],
        'He I': [4472, 5876, 6678, 7065],
        'Ca II': [np.mean([3934, 3969]), np.mean([8498, 8542, 8662])],
        'Fe II': [5018, 5169]}
    neb_lines = {'[O I]': [np.mean([6300, 6364])],
        'O I': [7774],
        '[Ca II]': [np.mean([7291,7324])],
        'H I': [4341, 4861, 6563],
        'He I': [5876, 6678, 7065],
        'Na I': [np.mean([5890, 5896])],
        'Mg I]':[4571]}
    
    colors = ['r','c','m','y','k','b','g','r','c','m']
    
    ###############################################################
    
    def __init__(self, f, v=12000, edge_width=100.0, smoothing_factor=0.02, verbose=True):
        ### Define some instance variables ###
        self.color_counter = 0
        self.line_counter = 0
        self.v = v
        # the default window size used to determine edges of lines
        self.edge_width = edge_width
        # defines the default spline smoothing factor: s = smoothing_factor * len(fl_array)
        self.smoothing_factor = smoothing_factor
        self.verbose = verbose
        # open the file
        try:
            self.wl,self.fl,self.er = np.loadtxt(f,unpack=True)
        except:
            self.wl,self.fl = np.loadtxt(f,unpack=True)
            self.er = None
        # pull out the object name, matching ptf or IAU filename
        self.name = re.search( '([sS][nN]\d{4}[a-zA-Z]+)|([Pp][Tt][Ff]\d{2}[a-zA-Z]+)', f).group() 
        self.datestring = re.search('\d{8}\.?\d*', f).group()
        self.fitted_lines = {} # keeps track of line properties
        self.plotted_lines = {} # keeps track of what's on the plot
        
        ### Run the fitting routine ###
        self.simple_plot()
        # self.add_lines(self.phot_lines)
        self.fit_lines()
        self.interact()
    
    ###############################################################
    
    def simple_plot(self):
        self.fig = pretty_plot_spectra(self.wl, self.fl, self.er)
        self.ymin, self.ymax = plt.axis()[2:]
        plt.title(self.name)
    
    def fit_lines(self, lines=None, edge_width=None, smoothing_factor=None, v=None):
        '''
        Fit for the properties of spectral lines.
         <lines> must be a dictionary with strings for keys and a list
          of central wavelengths as objects
         <edge_width> is the size of the edge window used to determine the continuum
         <smoothing_factor>: 0 to interpolate directly, len(spectra) is default
        '''
        if lines == None:
            lines = self.phot_lines
        if edge_width == None:
            edge_width = self.edge_width
        if smoothing_factor == None:
            smoothing_factor = self.smoothing_factor
        if v == None:
            v = self.v
        for ion in lines.keys():
            for i,wl in enumerate(lines[ion]):
                try:
                    if self.verbose: print 'fitting for',ion,wl
                    plot_lines = []
                    res = calc_everything(self.wl, self.fl, self.er, wl - wl*(v/3e5), 
                                          plot=1, width=edge_width, spline_smooth=smoothing_factor,
                                          line_container=plot_lines)
                    self.line_counter +=1
                    line_dict = {'ion':ion, 'wl':wl, 'edge_width':edge_width,
                                 'smoothing_factor':smoothing_factor,
                                 'pEW':res[0], 'wl_min':res[1],
                                 'rel_depth':res[2], 'FWHM':res[3]}
                    self.fitted_lines[self.line_counter] = line_dict
                    self.add_lines({'%d - '%self.line_counter+ion : res[1]}, v=0.0, line_container=plot_lines)
                    plot_lines.append(plt.annotate(str(self.line_counter), (res[1]+10,self.ymax-1), color='k', size=16))
                    self.plotted_lines[self.line_counter] = plot_lines
                except:
                    if self.verbose: print 'no',ion,'line at',wl
                    continue

    def add_lines(self, lines, v=None, line_container=None):
        '''
        Add vertical lines showing spectral line wavelengths at blueshifted velocity v.
        <lines> can be a string or a list of strings, matching one of the lines in the simple_lines
         dictionary, or a dictionary of lines (with strings for keys). <line_container> can
         be a list that will include all the line objects added to the plot, so you can remove
         them later.
        '''
        if v == None:
            v = self.v
        if type(lines) == str:
            wls = self.simple_lines[lines]
            wls = wls - wls*(v/3e5)
            c = self.colors[self.color_counter%len(self.colors)]
            self.color_counter +=1
            ls = plt.vlines( wls, self.ymin, self.ymax, linestyles='dotted', label=ion, color=c, lw=2 )
        elif type(lines) == list:
            for ion in lines:
                wls = self.simple_lines[ion]
                wls = wls - wls*(v/3e5)
                c = self.colors[self.color_counter%len(self.colors)]
                self.color_counter +=1
                ls = plt.vlines( wls, self.ymin, self.ymax, linestyles='dotted', label=ion, color=c, lw=2 )
        elif type(lines) == dict:
            for ion in lines.keys():
                wls = lines[ion]
                wls = wls - wls*(v/3e5)
                c = self.colors[self.color_counter%len(self.colors)]
                self.color_counter +=1
                ls = plt.vlines( wls, self.ymin, self.ymax, linestyles='dotted', label=ion, color=c, lw=2 )
        if line_container != None:
            line_container.append(ls)
        leg = plt.legend(fancybox=True, loc='best' )
        leg.get_frame().set_alpha(0.0)
        plt.draw()
    
    def delete_line(self, line_number):
        '''
        Remove a fitted line from the plot.
        '''
        if line_number not in self.plotted_lines.keys():
            print 'line',line_number,'not found'
            return
        else:
            _ = self.fitted_lines.pop(line_number)
            ls = self.plotted_lines.pop(line_number)
            for l in ls:
                l.remove()
        leg = plt.legend(fancybox=True, loc='best')
        leg.get_frame().set_alpha(0.0)
        plt.draw()    
    
    def save(self, fname=None, totext=True):
        if fname == None:
            fname = self.name + '_' + self.datestring + '.lines'
        out_lines = []
        for key in self.fitted_lines.keys():
            out_lines.append(self.fitted_lines[key])
        if totext:
            cw = 10      # character width to justify to
            fmt = '%.3f' # how to format the floats
            header = '# Line properties measured for ' + self.name + ' -- ' + self.datestring + '\n'
            header += '# ion'.ljust(cw) + 'wl'.ljust(cw) + 'wl_min'.ljust(cw) +\
                     'FWHM'.ljust(cw) + 'pEW'.ljust(cw) + 'rel_depth'.ljust(cw) + '\n'
            fout = open(fname+'.txt','w')
            fout.write(header)
            for l in out_lines:
                s = ('%s'%l['ion']).ljust(cw) + (fmt%l['wl']).ljust(cw) +\
                    (fmt%l['wl_min']).ljust(cw) + (fmt%l['FWHM']).ljust(cw) +\
                    (fmt%l['pEW']).ljust(cw) + (fmt%l['rel_depth']).ljust(cw) + '\n'
                fout.write(s)
            fout.close()
        else:
            pickle.dump(out_lines, open(fname+'.p','w'))
        if self.verbose: print 'saved line properties'
        
    def interact(self):
        '''
        Run the interactive script.
        '''
        while True:
            inn = raw_input('\n\nOptions:\n enter any line number to edit or\n'+\
                            ' n: add a new line\n s: save\n q: quit\n')
            try:
                line_number = int(re.search('\d+',inn).group())
            except AttributeError:
                if 'n' in inn.lower():
                    inn = raw_input('Enter the new ion name [optional: wavelength]\n')
                    try:
                        ion = re.search('[a-zA-Z]+\s*[a-zA-Z]*', inn).group()
                        try:
                            wl = float(re.search('\d+\.?\d*', inn).group())
                        except:
                            print 'Click once near center of new line'
                            wl = plt.ginput(timeout=60)[0][0]
                    except:
                        print 'what?\n'
                        continue
                    self.fit_lines({ion:[wl]}, v=0.0)
                    continue
                elif 's' in inn.lower():
                    self.save()
                    continue
                elif 'q' in inn.lower():
                    break
                else:
                    print 'what?\n'
                    continue
            if line_number not in self.fitted_lines.keys():
                print 'line',line_number,'not found'
                continue
            else:
                inn = raw_input('Options:\n d: delete this line\n'+\
                                ' w: adjust edge width and re-fit line\n'+\
                                ' s: adjust the smoothing parameter and re-fit line\n'+\
                                ' q: back to main\n')
                if 'd' in inn.lower():
                    if self.verbose: print 'dropping line',line_number
                    self.delete_line(line_number)
                    continue
                elif 'w' in inn.lower():
                    print 'current width:',self.fitted_lines[line_number]['edge_width']
                    inn = raw_input('Enter a new width:\n')
                    try:
                        new_width = float(inn)
                    except:
                        print 'what?\n'
                        continue
                    if self.verbose: print 'refitting line with width =',new_width
                    wl = self.fitted_lines[line_number]['wl']
                    ion = self.fitted_lines[line_number]['ion']
                    s = self.fitted_lines[line_number]['smoothing_factor']
                    self.delete_line(line_number)
                    self.fit_lines({ion:[wl]}, edge_width=new_width, smoothing_factor=s)
                elif 's' in inn.lower():
                    print 'current smoothing factor:',self.fitted_lines[line_number]['smoothing_factor']
                    inn = raw_input('Enter a new smoothing factor:\n')
                    try:
                        if 'none' in inn.lower():
                            new_s = None
                        else:
                            new_s = float(inn)
                    except:
                        print 'what?\n'
                        continue
                    if self.verbose: print 'refitting line with s =',new_s
                    wl = self.fitted_lines[line_number]['wl']
                    ion = self.fitted_lines[line_number]['ion']
                    e = self.fitted_lines[line_number]['edge_width']
                    self.delete_line(line_number)
                    self.fit_lines({ion:[wl]}, edge_width=e, smoothing_factor=new_s)
                elif 'q' in inn.lower():
                    continue
                
