"""

A quick script that scans a *.tex file and searches for all of the citations within
 it.  For any citation that is formatted with an ADS-compatible bibstring
 (i.e. 2013ApJ...768L..14P, or 2011MNRAS.412.1441L), this script will pull the
 bibtex entry from ADS and will insert them all into a .bib file.

-I Shivvers, 2015
"""

import re
import urllib2

def pull_all_citations( tex_file, bib_file, verbose=True, update=True ):
    """
    Given a .tex file, will scan it and search for all citations
    within it.  For any citation that is formatted with an ADS-compatible bibstring
    (i.e. 2013ApJ...768L..14P, or 2011MNRAS.412.1441L), this script will pull the
    bibtex entry from ADS and will insert them all into the .bib file given.
    WARNING: will overwrite the bib_file!
    If path to bib file is given, and file exists, and update=True, will scan it first and won't 
     bother to re-download citations.
    """
    uri = "http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=%s&data_type=BIBTEX"
    
    if update:
        try:
            oldbib = open( bib_file, 'r' ).read()
            oldcites = re.findall('@[^{]+{[^}]+\n', oldbib)
            oldcites = [c.split('{')[1].split(',')[0] for c in oldcites]
            print 'Already have:'
            print '\n'.join( [cc for cc in oldcites] ) 
        except:
            oldcites = []
    else:
        oldcites = []

    # first parse the .tex file and pull out all the citations
    fstring = open( tex_file, 'r' ).read()
    matches = re.findall( r'\\cite[pt].*{[^}]+}', fstring )
    citations = []
    for c in matches:
        c = c.split('{')[1].split('}')[0]
        for cc in c.split(','):
            if cc not in citations and cc not in oldcites:
                citations.append(cc)
    if verbose:
        print 'Found these new citations:'
        print '\n'.join( [cc for cc in citations] ) 

    # now go through each and try and pull the bibtex entry
    outf = open( bib_file, 'a' )
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

def drop_all_duplicates( bibfile, outfile=None, verbose=False ):
    """
    Parses a natbib style bibliography file (*.bib) and drops all
     entries with duplicate keywords.
    If outfile==None, overwrites input file.
    If verbose==True, says everything it would drop.
    """
    if outfile == None:
        outfile = bibfile
    instring = open(bibfile, 'r').read()
    outf = open(outfile, 'w')
    keywords = []
    inlevel = 0
    keeper = False
    for i,char in enumerate(instring):
        if char == '{':
            inlevel +=1
        elif char == '}':
            inlevel -=1
        elif (inlevel==0) & (char == '@'):
            keyword = instring[i:].split(',',1)[0].split('{')[1]
            if keyword not in keywords:
                keywords.append(keyword)
                keeper = True
            else:
                keeper = False
                if verbose:
                    print 'dropping',keyword
        if keeper:
            outf.write( char )
    outf.close()
    return keywords


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