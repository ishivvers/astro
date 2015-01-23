#!/usr/bin/env python
"""
A command-line program that produces a live plot of the progress
 of a SYNAPPS task.  Requires the path to the yaml of the running
 SYNAPPS instance and path to a running log of the SYNAPPS process
 as produced, for example by the following commands:
 ::: synapps SN2011dh.yaml > synapps.log
 ::: synapps sn2011dh.yaml | tee synapps.log
Usage: synwatch <synapps.yaml.file> <synapps.running.log> <refresh interval (minutes)>

"""

try:
    from pyES import Common, Synpp, Synapps
except ImportError:
    from astro.pyES import Common, Synpp, Synapps
from subprocess import Popen, PIPE
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


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

def set_OMP():
    '''
    Manage the setting of OMP_NUM_THREADS to most efficiently
    utilize parallelization.
    '''
    nproc, e = Popen("nproc --all",shell=True,stdout=PIPE).communicate()
    os.environ['OMP_NUM_THREADS'] = nproc

def parse_log( yaml_file, log_file, nlines=3, stepsize=1 ):
    '''
    Parse a yaml and a log, returning a SYN++ input yaml as a string as
     well as several other useful things.
    nlines: the number of recent models to report back
    stepsize: if stepsize > 1, will skip <stepsize> models in between reported models
    '''
    
    # Parse the log to find most recent minimum
    last_min = None
    last_mins = []
    for line in open( log_file, "r" ).readlines() :
        try:
            if line.startswith( "New Min" ) or line.startswith( "Final Min" ) :
                last_min = line.rstrip()
                last_mins.append(last_min)
                if len(last_mins) > nlines*stepsize:
                    last_mins.pop(0)
        except:
            continue
    if not last_min :
        raise Exception("ERROR: No 'New Min' or 'Final Min' lines found in log: %s" % log_file)
    last_mins.reverse()
    last_mins = last_mins[::stepsize]
    for i,lm in enumerate(last_mins):
        last_mins[i] = [ float( x ) for x in lm[ lm.find( "[" ) + 1 : lm.find( "]" ) ].split() ]
    
    setups = []
    for last_min in last_mins:
        # Create a synapps object
        synapps = Synapps.Synapps.create( yaml.load( open( yaml_file, "r" ) ) )
        
        # Is the log compatible with the YAML control file?
        num_ions = 0
        num_active_ions = 0
        for ion in synapps.config.ions :
            num_ions += 1
            num_active_ions += 1 if ion.active else 0
        num_params = num_active_ions * 5
        num_params += 6
        if num_params != len( last_min ) :
            raise Exception("ERROR: Incompatible synapps.yaml and synapps.log: %d and %d parameters" % ( num_params, len( last_min ) ))
        
        # Substitute log entries into synapps control object's start slots.
        synapps.config.a0.start      = last_min.pop( 0 )
        synapps.config.a1.start      = last_min.pop( 0 )
        synapps.config.a2.start      = last_min.pop( 0 )
        synapps.config.v_phot.start  = last_min.pop( 0 )
        synapps.config.v_outer.start = last_min.pop( 0 )
        synapps.config.t_phot.start  = last_min.pop( 0 )
        
        j = 0
        for i in range( num_ions ) :
            if not synapps.config.ions[ i ].active :
                continue
            synapps.config.ions[ i ].log_tau.start = last_min[ j + 0 * num_active_ions ]
            synapps.config.ions[ i ].v_min.start   = last_min[ j + 1 * num_active_ions ]
            synapps.config.ions[ i ].v_max.start   = last_min[ j + 2 * num_active_ions ]
            synapps.config.ions[ i ].aux.start     = last_min[ j + 3 * num_active_ions ]
            synapps.config.ions[ i ].temp.start    = last_min[ j + 4 * num_active_ions ]
            j += 1

        # Create a syn++ yaml control file.
        output   = Synpp.Output.create()
        grid     = synapps.grid
        opacity  = synapps.opacity
        source   = synapps.source
        spectrum = synapps.spectrum

        setup = Synpp.Setup.create()
        setup.a0      = synapps.config.a0.start
        setup.a1      = synapps.config.a1.start
        setup.a2      = synapps.config.a2.start
        setup.v_phot  = synapps.config.v_phot.start
        setup.v_outer = synapps.config.v_outer.start
        setup.t_phot  = synapps.config.t_phot.start
        for i, ion in enumerate( synapps.config.ions ) :
            if synapps.config.ions[ i ].active:
                setup.ions.append( Synpp.Ion( ion.ion, ion.active, ion.log_tau.start, ion.v_min.start, ion.v_max.start, ion.aux.start, ion.temp.start ) )    
        setups.append(setup)
        
    return str(Synpp.Synpp( output, grid, opacity, source, spectrum, [setups[0]] )), setups, synapps.evaluator.target_file


def take_snapshot( yaml_file, log_file, verbose=False, tmpfile='synwatch.syn++.yaml', cmd='syn++' ):
    '''
    Take a snapshot from the logfile and return the spectra and the synapps object.
    '''
    if verbose: print 'scanning logfile...'
    yaml_string, setups, target_file = parse_log(yaml_file,log_file, stepsize=100)
    open(tmpfile,'w').write( yaml_string )
    if verbose: print 'running syn++'
    res = Popen( cmd+' '+tmpfile, shell=True, stderr=PIPE, stdout=PIPE )
    o,e = res.communicate()
    Popen( 'rm '+tmpfile, shell=True )
    synthetic_spec = np.array([ map(float,[l.split(' ')[0],l.split(' ')[1]]) for l in o.strip().split('\n') ])
    true_spec = np.loadtxt( target_file )
    return synthetic_spec, true_spec, setups


def update_plot( syn_spec, true_spec, setups, name='SYNAPPS' ):
    '''
    Take the results of a snapshot and update the current plot.
    Plot sketch: large panel with both spectra (include error)
                 large panel below with rows of bar plots:
                   log_tau/aux (double-axis bar)
                   v_min/v_max (double-ended bar with ends at v_phot, v_outer)
                   tmp (bar with hline for t_phot)
                 right panel spanning all, with:
                   summary info (file names, elapsed time, number of fits)
                   a0,a1,a2 bar plot
    '''
    plt.clf()
    # first the spectral plot
    ax = plt.subplot2grid( (6,4), (0,0), rowspan=2, colspan=4 )
    ax.fill_between( true_spec[:,0], true_spec[:,1]+true_spec[:,2],
                     true_spec[:,1]-true_spec[:,2], interpolate=True, color='grey', alpha=0.1 )
    ax.plot( true_spec[:,0], true_spec[:,1], color='k' )
    ax.plot( syn_spec[:,0], syn_spec[:,1], color='r' )
    a_str = 'a0: {}\na1: {}\na2: {}'.format( setups[0].a0, setups[0].a1, setups[0].a2 )
    ax.annotate( a_str, (.8, .8), xycoords='figure fraction' )
    ax.set_title( name )
    ax.set_yticks([])
    
    width = 0.3
    offsets = [0.0, 0.3, 0.6]
    alphas = [1.0, 0.5, 0.25]
    axes = [plt.subplot2grid( (6,4), (2,0), colspan=4 ), plt.subplot2grid( (6,4), (3,0), colspan=4 ),
            plt.subplot2grid( (6,4), (4,0), colspan=4 ), plt.subplot2grid( (6,4), (5,0), colspan=4 )]
    for ijk, setup in enumerate(setups):
        # the log_tau plot
        ax = axes[0]
        x = np.arange(len(setup.ions))
        log_tau = [i.log_tau for i in setup.ions]
        ax.bar( x+offsets[ijk], log_tau, color='b', alpha=alphas[ijk], width=width )
        ax.hlines( 0.0, x[0], x[-1]+1, linestyles='dashed' )
        ax.set_ylabel(r'Log $\tau$')
        ax.set_xticks([])
    
        # now the velocities plot
        ax = axes[1]
        v_maxs = [i.v_max for i in setup.ions]
        v_mins = [i.v_min for i in setup.ions]
        ax.bar( x+offsets[ijk], v_maxs, color='red', width=width, alpha=alphas[ijk] )
        ax.bar( x+offsets[ijk], v_mins, color='white', edgecolor='white', width=width )
        ax.set_ylim( setup.v_phot, setup.v_outer )
        ax.set_ylabel(r'V (km/s)')
        ax.set_xticks([])
    
        # now the exponents plot
        ax = axes[2]
        exponents = [i.aux for i in setup.ions]
        ax.bar( x+offsets[ijk], exponents, width=width, color='blue', alpha=alphas[ijk] )
        ax.set_ylabel(r'Exponent')
        ax.set_xticks([])    
    
        # now the temperatures
        ax = axes[3]
        temps = [i.temp for i in setup.ions]
        ax.hlines( setup.t_phot, x[0], x[-1]+1, linestyles='dashed' )
        ax.bar( x+offsets[ijk], temps, color='red', width=width, alpha=alphas[ijk] )
        ax.set_ylabel(r'T (kK)')
        labels = [ion_dict[i.ion] for i in setup.ions]
        ax.set_xticks( x+0.4 )
        ax.set_xticklabels( labels )
        ax.annotate('photosphere', (x[-1], setup.t_phot+0.5))
    plt.draw()
    plt.show()
    


if __name__ == '__main__':
    try:
        assert( os.path.isfile(sys.argv[-3]) )
        assert( os.path.isfile(sys.argv[-2]) )
        refresh_interval = float( sys.argv[-1] ) * 60
    except:
        raise Exception('Usage: synwatch <synapps.yaml.file> <synapps.running.log> <refresh interval (minutes)>')
    # plt.ion()
    set_OMP()
    fig = plt.figure( figsize=(12,8) )
    nframe = 0
    while True:
        update_plot( *take_snapshot( sys.argv[-3], sys.argv[-2], True ), name=sys.argv[-3]+' - '+str(nframe) )
        nframe +=1
        print '{}: sleeping for {} minutes'.format(nframe, sys.argv[-1])
        sleep( refresh_interval )
    
