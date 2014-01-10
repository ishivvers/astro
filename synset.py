#!/usr/bin/env python
"""
An interactive program that provides a gui to set the values of SYN++
 parameters.
Usage: synset <yaml_file_path> <comparision_spectrum_file>

"""
import os
import sys
import yaml
import mechanize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import colors
from subprocess import Popen, PIPE
from pyES import Common, Synpp

ion_dict = {100:'H I', 200:'He I', 201:'He II', 300:'Li I', 301:'Li II', 400:'Be I',
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


class SynSet():
    '''
    A class to handle all interactions between a human and a SYN++ file.
    '''
    def __init__( self, yaml_file, spec_file, synp_cmd='syn++', verbose=True, gui_debug=False ):
        '''
        Create a set of sliders that interacts with a syn++ yaml
         file saved at the path <yaml_file>, and compares to a 
         spectrum file (Flipper format) saved at the 
         path <spec_file>
        '''
        # load up the synpp object, the comparision spectrum, and calculate a first syn++ spectrum
        self.syn = Synpp.Synpp.create( yaml.load( open(yaml_file,"r") ) )
        self.spec = np.loadtxt( spec_file )
        self.synp_cmd = synp_cmd
        self.yaml_file = yaml_file
        self.verbose = verbose
        self.gui_debug = gui_debug
        self.ion_lines = {}
        self.nproc = None
        # set the number of threads to use with synapps
        self.handle_OMP()
        if self.verbose: print 'running syn++'
        if not self.gui_debug:
            res = Popen( synp_cmd+' '+yaml_file, shell=True, stderr=PIPE, stdout=PIPE )
            o,e = res.communicate()
            if e:
                raise Exception('Error running SYN++: '+e)
            self.synspec = np.array([ map(float,[l.split(' ')[0],l.split(' ')[1]]) for l in o.strip().split('\n') ])
        # build up the figure and go
        self._build_figure(newfig=True, current_calc=True)

    def close(self):
        # restore the number of threads to the previous value
        self.handle_OMP()
    
    def handle_OMP(self):
        '''
        Manage the setting of OMP_NUM_THREADS to most efficiently
        utilize parallelization.
        '''
        if self.verbose: print 'handling OMP_NUM_THREADS bash variable'
        if self.nproc == None:
            # on init, the variable is not set
            self.nproc, e = Popen("nproc --all",shell=True,stdout=PIPE).communicate()
            try:
                self.OMP = os.environ['OMP_NUM_THREADS']
            except KeyError:
                self.OMP = None
            os.environ['OMP_NUM_THREADS'] = self.nproc
        else:
            if self.OMP != None:
                # restore the bash variable to its original state
                os.environ['OMP_NUM_THREADS'] = self.OMP


    def _build_figure(self, newfig=False, current_calc=False):
        if self.verbose: print 'creating figure'
        # the spectral plot
        if newfig:
            self.fig = plt.figure(figsize=(14,10))
            plt.subplots_adjust(wspace=0.3)
        else:
            plt.clf()
        ntall = 4 + len(self.syn.setups[0].ions)
        self.spec_ax = plt.subplot2grid( (ntall, 6), (0,0), rowspan=3, colspan=4 )
        self.spec_ax.plot( self.spec[:,0], self.spec[:,1], color='k' )
        if not self.gui_debug:
            self.model_line, = self.spec_ax.plot( self.synspec[:,0], self.synspec[:,1], color='r', label='Requires Update' )
            self.ax_legend = self.spec_ax.legend(loc=1)
            if current_calc: self.ax_legend.set_visible(False)
        self.spec_ax.set_yticks([])
        self.spec_ax.get_xaxis().set_ticks_position('top')
        # the a* sliders
        a_axes = [ plt.subplot2grid( (ntall, 6), (0,4) ),
                   plt.subplot2grid( (ntall, 6), (1,4) ),
                   plt.subplot2grid( (ntall, 6), (2,4) )]
        self.a_sliders = [ Slider(a_axes[0], 'a0', self.syn.setups[0].a0-5.0, self.syn.setups[0].a0+5.0,
                                  valinit=self.syn.setups[0].a0, closedmin=True, closedmax=True, valfmt='%1.1f' ),
                           Slider(a_axes[1], 'a1', self.syn.setups[0].a1-5.0, self.syn.setups[0].a1+5.0,
                                  valinit=self.syn.setups[0].a1, closedmin=True, closedmax=True, valfmt='%1.1f' ),
                           Slider(a_axes[2], 'a2', self.syn.setups[0].a2-5.0, self.syn.setups[0].a2+5.0,
                                  valinit=self.syn.setups[0].a2, closedmin=True, closedmax=True, valfmt='%1.1f' )]
        for s in self.a_sliders:
            s.on_changed(self._update_as)
            s.label.set_weight('bold')
            s.label.set_color('grey')
            s.label.set_x(0.5)
            s.label.set_y(0.8)
        # the other global siders
        o_axes = [ plt.subplot2grid( (ntall, 6), (0,5) ),
                   plt.subplot2grid( (ntall, 6), (1,5) ),
                   plt.subplot2grid( (ntall, 6), (2,5) )]
        self.o_sliders = [ Slider(o_axes[0], 'v_phot', self.syn.setups[0].v_phot-5.0, self.syn.setups[0].v_phot+5.0,
                                  valinit=self.syn.setups[0].v_phot, closedmin=True, closedmax=True, valfmt='%1.1f' ),
                           Slider(o_axes[1], 'v_outer', self.syn.setups[0].v_outer-5.0, self.syn.setups[0].v_outer+5.0,
                                  valinit=self.syn.setups[0].v_outer, closedmin=True, closedmax=True, valfmt='%1.1f' ),
                           Slider(o_axes[2], 't_phot', self.syn.setups[0].t_phot-5.0, self.syn.setups[0].t_phot+5.0,
                                  valinit=self.syn.setups[0].t_phot, closedmin=True, closedmax=True, valfmt='%1.1f' )]
        for s in self.o_sliders:
            s.on_changed(self._update_os)
            s.label.set_weight('bold')
            s.label.set_color('grey')
            s.label.set_x(0.5)
            s.label.set_y(0.8)
        # the properties of each ion
        self.active_buttons = []
        self.tau_sliders = []
        self.vmin_sliders = []
        self.vmax_sliders = []
        self.aux_sliders = []
        self.temp_sliders = []
        for i,ion in enumerate(self.syn.setups[0].ions):
            # active radio button
            ax = plt.subplot2grid( (ntall, 6), (3+i,0) )
            button = CheckButtons(ax, [ion_dict[ion.ion]], [ion.active])
            button.on_clicked(self._update_actives)
            self.active_buttons.append(button)
            # log tau slider
            ax = plt.subplot2grid( (ntall, 6), (3+i,1) )
            slider = Slider(ax, 'log_tau', ion.log_tau-5.0, ion.log_tau+5, 
                            valinit=ion.log_tau, closedmin=True, closedmax=True, valfmt='%1.1f')
            slider.on_changed(self._update_ions)
            slider.label.set_weight('bold')
            slider.label.set_color('grey')
            slider.label.set_x(0.5)
            slider.label.set_y(0.8)
            self.tau_sliders.append(slider)
            # vmin slider
            ax = plt.subplot2grid( (ntall, 6), (3+i,2) )
            slider = Slider(ax, 'v_min', ion.v_min-5.0, ion.v_min+5, valinit=ion.v_min,
                            closedmin=True, closedmax=True, valfmt='%1.1f',
                            slidermin=self.o_sliders[0])
            slider.on_changed(self._update_ions)
            slider.label.set_weight('bold')
            slider.label.set_color('grey')
            slider.label.set_x(0.5)
            slider.label.set_y(0.8)
            self.vmin_sliders.append(slider)
            # vmax slider
            ax = plt.subplot2grid( (ntall, 6), (3+i,3) )
            slider = Slider(ax, 'v_max', ion.v_max-5.0, ion.v_max+5, valinit=ion.v_max,
                            closedmin=True, closedmax=True, valfmt='%1.1f',
                            slidermax=self.o_sliders[1])
            slider.on_changed(self._update_ions)
            slider.label.set_weight('bold')
            slider.label.set_color('grey')
            slider.label.set_x(0.5)
            slider.label.set_y(0.8)
            self.vmax_sliders.append(slider)
            # aux slider
            ax = plt.subplot2grid( (ntall, 6), (3+i,4) )
            slider = Slider(ax, 'aux', ion.aux-5.0, ion.aux+5, 
                            valinit=ion.aux, closedmin=True, closedmax=True, valfmt='%1.1f')
            slider.on_changed(self._update_ions)
            slider.label.set_weight('bold')
            slider.label.set_color('grey')
            slider.label.set_x(0.5)
            slider.label.set_y(0.8)
            self.aux_sliders.append(slider)
            # temp slider
            ax = plt.subplot2grid( (ntall, 6), (3+i,5) )
            slider = Slider(ax, 'temp', ion.temp-5.0, ion.temp+5, 
                            valinit=ion.temp, closedmin=True, closedmax=True, valfmt='%1.1f')
            slider.on_changed(self._update_ions)
            slider.label.set_weight('bold')
            slider.label.set_color('grey')
            slider.label.set_x(0.5)
            slider.label.set_y(0.8)
            self.temp_sliders.append(slider)
        # the buttons
        # ax = plt.subplot2grid( (ntall,6), (ntall-1,3) )
        # self.rezero_button = Button(ax, 'ReZero', color='grey', hovercolor='green')
        # self.rezero_button.on_clicked(self._rezero)
        ax = plt.subplot2grid( (ntall,6), (ntall-1,0) )
        self.tinput_button = Button(ax, 'Text CMD', color='grey', hovercolor='green')
        self.tinput_button.on_clicked(self._accept_input)
        ax = plt.subplot2grid( (ntall,6), (ntall-1,4) )
        self.calc_button = Button(ax, 'Calc', color='grey', hovercolor='green')
        self.calc_button.on_clicked(self._calculate)
        ax = plt.subplot2grid( (ntall,6), (ntall-1,5) )
        self.save_button = Button(ax, 'Save', color='grey', hovercolor='green')
        self.save_button.on_clicked(self._save)

    def _replot_spectrum(self):
        if self.verbose: print 'replotting model'
        if not self.gui_debug:
            self.model_line.remove()
            self.model_line, = self.spec_ax.plot( self.synspec[:,0], self.synspec[:,1], color='r', label='Requires Update' )
            self.ax_legend = self.spec_ax.legend(loc=1)
            self.ax_legend.set_visible(False)
        self.spec_ax.set_yticks([])
        plt.draw()

    def _rezero(self, val):
        plt.close()
        self._build_figure(newfig=True, current_calc=False)

    def _update_as(self, val):
        self.syn.setups[0].a0 = self.a_sliders[0].val
        self.syn.setups[0].a1 = self.a_sliders[1].val
        self.syn.setups[0].a2 = self.a_sliders[2].val
        if not self.gui_debug: self.ax_legend.set_visible(True)
        plt.draw()

    def _update_os(self, val):
        self.syn.setups[0].v_phot = self.o_sliders[0].val
        self.syn.setups[0].v_outer = self.o_sliders[1].val
        self.syn.setups[0].t_phot = self.o_sliders[2].val
        if not self.gui_debug: self.ax_legend.set_visible(True)
        plt.draw()

    def _update_actives(self, val):
        for i,ion in enumerate(self.syn.setups[0].ions):
            ion.active = self.active_buttons[i].lines[0][0].get_visible()
        if not self.gui_debug: self.ax_legend.set_visible(True)
        plt.draw()

    def _update_ions(self, val):
        for i,ion in enumerate(self.syn.setups[0].ions):
            ion.log_tau = self.tau_sliders[i].val
            ion.v_min = self.vmin_sliders[i].val
            ion.v_max = self.vmax_sliders[i].val
            ion.aux = self.aux_sliders[i].val
            ion.temp = self.temp_sliders[i].val
        if not self.gui_debug: self.ax_legend.set_visible(True)
        plt.draw()

    def _calculate(self, event, tmpfile='synset.yaml'):
        if self.verbose: print 'running syn++'
        open(tmpfile,'w').write( str(self.syn) )
        res = Popen( self.synp_cmd+' '+tmpfile, shell=True, stderr=PIPE, stdout=PIPE )
        o,e = res.communicate()
        if e:
            raise Exception('Error running SYN++: '+e)
        Popen( 'rm '+tmpfile, shell=True )
        self.synspec = np.array([ map(float,[l.split(' ')[0],l.split(' ')[1]]) for l in o.strip().split('\n') ])
        # replot the spectra
        self._replot_spectrum()

    def _accept_input(self, event):
        print 'Enter a command. "h" for help, "q" to exit the cmd line.\n'
        res = raw_input()
        if res == 'h':
            print 'Ion Lines: enter an ion name (i.e. "Fe I") to'
            print ' add or remove the strongest lines of that ion from'
            print ' the plot. Enter "clear" to remove all ion lines'
            print ' from the plot.'
            print ' Include a blue shift velocity (in km/s) as a second argument'
            print ' to plot the lines at that blue shift.'
            print 'Other: enter any other valid python command to execute it.\n'
            self._accept_input(None)
        elif res == 'q':
            pass
        elif res == 'clear':
            self._remove_lines( self.ion_lines.keys() )
        elif ' '.join(res.split(' ')[:2]) in ion_dict.values():
            ion = ' '.join(res.split(' ')[:2])
            if ion in self.ion_lines.keys():
                self._remove_lines(ion)
            else:
                try:
                    v = float(res.split(' ')[2])
                except:
                    v = 0.0
                self._add_lines(ion, v)
        else:
            try:
                exec(res)
            except:
                print 'You entered:',res
                print 'Interpreted input as command. Failure.\n'
                self._accept_input(None)
    
    def _add_lines(self, ion, v=0.0):
        if self.verbose: print 'querying NIST for',ion,'lines'
        wls, aks = query_nist(ion, self.syn.output.min_wl, self.syn.output.max_wl)
        ymin,ymax = self.spec_ax.axis()[2:]
        color = colors.cnames.keys()[ np.random.randint(len(colors.cnames)) ]
        if self.verbose: print 'adding',color,'lines for',ion,'at velocity',v
        wls = wls - wls*(v/3e5)
        self.ion_lines[ion] = self.spec_ax.vlines( wls, ymin, ymax, linestyles='dotted', color=color )
        plt.draw()
    
    def _remove_lines(self, ions):
        if self.verbose: print 'removing lines for',ions
        if type(ions) == list:
            # ions is a list of ions
            for ion in ions:
                self.ion_lines[ion].remove()
                self.ion_lines.pop(ion)
        else:
            # ions is a single ion
            self.ion_lines[ions].remove()
            self.ion_lines.pop(ions) 
        plt.draw()
        
    def _save(self, event):
        if self.verbose: print 'saving to file',self.yaml_file
        open(self.yaml_file, 'w').write(str(self.syn))


if __name__ == '__main__':
    try:
        assert( os.path.isfile(sys.argv[-1]) )
        assert( os.path.isfile(sys.argv[-2]) )
    except:
        raise Exception('Usage: synset <yaml_file_path> <comparision_spectrum_file>')
    
    S = SynSet( sys.argv[-2], sys.argv[-1] )
    plt.show()
    S.close()
