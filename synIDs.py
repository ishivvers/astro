#!/usr/bin/env python
"""
A script that produces a set of plots to indicate
 the effect of each individual ion in a syn++.yaml configuration.
Usage: synIDs <yaml_file_path> <comparision_spectrum_file> <off/on>
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from subprocess import Popen, PIPE
from pyES import Synpp
from astro.iAstro import pretty_plot_spectra

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


def show_one_off(yaml_file, spec_file, save_plots=True, save_path='ions_off_plots/', show_plots=False, skip=[], verbose=True):
    '''
    Produce a set of plots showing the importance of various ions in
     a syn++ configuration file by removing them one at a time.
    '''

    ### load up files ###
    original_syn = Synpp.Synpp.create( yaml.load( open(yaml_file,"r") ) )
    original_model = run_syn(original_syn)
    syn = deepcopy(original_syn)
    spec = np.loadtxt( spec_file )

    ### if we're saving output, make a directory for it ###
    if save_plots:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    ### set the number of threads to use with synapps ##
    if verbose: print 'handling OMP_NUM_THREADS bash variable'
    set_OMP()

    ### now cycle through ions ###
    for i,ion in enumerate(syn.setups[0].ions):
        # ignore any ions not turned on in configuration file
        if not original_syn.setups[0].ions[i].active:
            continue
        # ignore any we're told to skip
        if ion_dict[ion.ion] in skip:
            continue
        if verbose: print 'working on ion',ion_dict[ion.ion]
        # toggle this ion and run syn++
        ion.active = False
        model = run_syn(syn)
        ion.active = True
        # make a plot
        fig = pretty_plot_spectra(spec[:,0], spec[:,1], spec[:,2])
        plt.plot(original_model[:,0], original_model[:,1], 'cyan', lw=2, label='Full')
        plt.plot(model[:,0], model[:,1], 'red', lw=2, label='Without '+ion_dict[ion.ion])
        leg = plt.legend(loc='best')
        leg.get_frame().set_alpha(0.0)
        plt.title(yaml_file + ' -- Effect of '+ion_dict[ion.ion])
        if save_plots:
            if verbose: print 'saving image for',ion_dict[ion.ion]
            plt.savefig(save_path + yaml_file + '.' + ion_dict[ion.ion].replace(' ','_') + '.png')
        if show_plots:
            plt.show()
        plt.close()

def show_one_on(yaml_file, spec_file, save_plots=True, save_path='ions_on_plots/', show_plots=False, skip=[], verbose=True):
    '''
    Produce a set of plots showing the importance of various ions in
     a syn++ configuration file by including them one at a time.
    '''

    ### load up files ###
    original_syn = Synpp.Synpp.create( yaml.load( open(yaml_file,"r") ) )
    original_model = run_syn(original_syn)
    syn = deepcopy(original_syn)
    spec = np.loadtxt( spec_file )
    for ion in syn.setups[0].ions:
        ion.active = False

    ### if we're saving output, make a directory for it ###
    if save_plots:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    ### set the number of threads to use with synapps ##
    if verbose: print 'handling OMP_NUM_THREADS bash variable'
    set_OMP()

    ### now cycle through ions ###
    for i,ion in enumerate(syn.setups[0].ions):
        # ignore any ions not turned on in configuration file
        if not original_syn.setups[0].ions[i].active:
            continue
        # ignore any we're told to skip
        if ion_dict[ion.ion] in skip:
            continue
        if verbose: print 'working on ion',ion_dict[ion.ion]
        # toggle this ion and run syn++
        ion.active = True
        model = run_syn(syn)
        ion.active = False
        # make a plot
        fig = pretty_plot_spectra(spec[:,0], spec[:,1], spec[:,2])
        plt.plot(original_model[:,0], original_model[:,1], 'cyan', lw=2, label='Full')
        plt.plot(model[:,0], model[:,1], 'red', lw=2, label='Only '+ion_dict[ion.ion])
        leg = plt.legend(loc='best')
        leg.get_frame().set_alpha(0.0)
        plt.title(yaml_file + ' -- Effect of '+ion_dict[ion.ion])
        if save_plots:
            if verbose: print 'saving image for',ion_dict[ion.ion]
            plt.savefig(save_path + yaml_file + '.' + ion_dict[ion.ion].replace(' ','_') + '.png')
        if show_plots:
            plt.show()
        plt.close()

def set_OMP():
    '''
    Manage the setting of OMP_NUM_THREADS to most efficiently
     utilize parallelization.
    '''
    nproc, e = Popen("nproc --all",shell=True,stdout=PIPE).communicate()
    os.environ['OMP_NUM_THREADS'] = nproc

def run_syn(syn, tmpfile='synIDs.tmp', synpp='syn++'):
    '''
    Run a syn++ configuration, returning a 2d array of wl, fl.
    '''
    open(tmpfile,'w').write( str(syn) )
    o,e = Popen( synpp+' '+tmpfile, shell=True, stderr=PIPE, stdout=PIPE ).communicate()
    os.remove(tmpfile)
    if e:
        raise Exception('Error running SYN++: '+e)
    return np.array([ map(float,[l.split(' ')[0],l.split(' ')[1]]) for l in o.strip().split('\n') ])



if __name__ == '__main__':
    try:
        assert( sys.argv[-1] in ['on','off'] )
        assert( os.path.isfile(sys.argv[-2]) )
        assert( os.path.isfile(sys.argv[-3]) )
    except:
        raise Exception('Usage: synIDs <yaml_file_path> <comparision_spectrum_file> <off/on>')

    if sys.argv[-1] == 'off':
        show_one_off(sys.argv[-3], sys.argv[-2])
    else:
        show_one_on(sys.argv[-3], sys.argv[-2])