"""

A quick script that scans a *.tex file and searches for all of the citations within
 it.  For any citation that is formatted with an ADS-compatible bibstring
 (i.e. 2013ApJ...768L..14P, or 2011MNRAS.412.1441L), this script will pull the
 bibtex entry from ADS and will insert them all into a .bib file.

-I Shivvers, 2015
"""

import re
import urllib2

def pull_all_citations( tex_file, bib_file, verbose=True ):
    """
    Given a .tex file, will scan it and search for all citations
    within it.  For any citation that is formatted with an ADS-compatible bibstring
    (i.e. 2013ApJ...768L..14P, or 2011MNRAS.412.1441L), this script will pull the
    bibtex entry from ADS and will insert them all into the .bib file given.
    WARNING: will overwrite the bib_file!
    """
    uri = "http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=%s&data_type=BIBTEX"

    # first parse the .tex file and pull out all the citations
    fstring = open( tex_file, 'r' ).read()
    matches = re.findall( '\\cite[pt]{[^\s]+}', fstring )
    citations = []
    for c in matches:
        c = c.split('{')[1].strip('}')
        for cc in c.split(','):
            citations.append(cc)

    # now go through each and try and pull the bibtex entry
    outf = open( bib_file, 'w' )
    for c in citations:
        if verbose:
            print 'Searching ADS for',c,'...',
        try:
            page = urllib2.urlopen( uri%c ).read()
        except urllib2.HTTPError:
            print 'FAILURE: could not find',c,'on ADS.'
            if verbose:
                print 'URI:  ',uri%c
            continue
        text = '@'+page.split('@')[1]
        outf.write('\n'+text+'\n')
        if verbose:
            print 'Success!'
    outf.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
    A quick script that scans a *.tex file and searches for all of the citations within
    it.  For any citation that is formatted with an ADS-compatible bibstring
    (i.e. 2013ApJ...768L..14P, or 2011MNRAS.412.1441L), this script will pull the
    bibtex entry from ADS and will insert them all into a .bib file.

    -I Shivvers, 2015
    ''')
    parser.add_argument('tex_file', metavar='LaTeX file', type=str,
                        help='path to input .tex file')
    parser.add_argument('bib_file', metavar='output BibTeX file', type=str,
                        help='name of output .bib file')
    parser.add_argument('-v', dest='verbose', action='store_true', default=False,
                        help='be verbose.')
    args = parser.parse_args()
    pull_all_citations( args.tex_file, args.bib_file, args.verbose )