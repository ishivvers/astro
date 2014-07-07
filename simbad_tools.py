"""
A quick library to deal with searching simbad for info
 about a SN and parsing the results.

Author: Isaac Shivvers, ishivvers@berkeley.edu, 2014


example SIMBAD uri query:
http://simbad.u-strasbg.fr/simbad/sim-id?output.format=ASCII&Ident=sn%201998S
"""

import re
from urllib2 import urlopen

def get_SN_info( name ):
    """
    Queries simbad for SN redshift and host galaxy.
    If redshift is not given for SN, attempts to resolve link to 
     host galaxy and report its redshift.
    Returns ( redshift, host_name, redshift_citation ), with
     values of None inserted whenever it cannot resolve the value.
    """
    simbad_uri = "http://simbad.u-strasbg.fr/simbad/sim-id?output.format=ASCII&Ident=%s"
    regex_redshift = "Redshift:\s+\d+\.\d+.+"
    regex_host = "apparent\s+host\s+galaxy\s+.+?\{(.*?)\}"

    result = urlopen( simbad_uri % name.replace(' ','%20') ).read()
    resred = re.search( regex_redshift, result )
    reshost = re.search( regex_host, result )

    try:
        redshift = float(resred.group().strip('Redshift: ').split(' ')[0])
        citation = resred.group().split(' ')[-1]
    except AttributeError:
        redshift = None
        citation = None
    
    try:
        host = reshost.group().split('{')[1].split('}')[0]
    except AttributeError:
        host = None

    if (redshift == None) and (host != None):
        # get the redshift from the host galaxy
        result = urlopen( simbad_uri % host.replace(' ','%20') ).read()
        resred = re.search( regex_redshift, result )
        try:
            redshift = float(resred.group().strip('Redshift: ').split(' ')[0])
            citation = resred.group().split(' ')[-1]
        except AttributeError:
            pass

    return (redshift, host, citation)