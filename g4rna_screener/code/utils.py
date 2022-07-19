#!/usr/bin/env python

#    Identification of potential RNA G-quadruplexes by G4RNA screener.
#    Copyright (C) 2018  Jean-Michel Garant
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import regex
import argparse
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict

class Formatter(argparse.HelpFormatter):
    """
    Extended HelpFormatter class in order to correct the greediness of --columns
    that includes the last positional argument. This extension of HelpFormatter
    brings the positional argument to the beginning of the command and the 
    optonal arguments are send to the end.
    
    This snippet of code was adapted from user "hpaulj" from StackOverflow.
    """
    # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage: '
        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)
        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups) # NEW
            usage = ' '.join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

def verbosify(verbose, message, flush=False):
    """
    Take care of the verbosity for the user.
    
    Supports both Boolean value of verbose and numerical level of verbose.
    Either print or flush message.
    """
    if verbose == False or verbose == 0:
        pass
    elif (verbose == True or verbose > 0) and flush == False:
        sys.stdout.write(message+"\n")
    elif flush == True:
        sys.stdout.write(message+"..."+" "*(77-len(message))+"\r")
        sys.stdout.flush()

def connect_psql(host, user, passwd, db, number_of_cursor):# schema,
    """
    Connects to a PostGreSQL database and generates cursors.
    
    Returns a list.
    [0] connection object
    [1] first cursor
    [2] second cursor
    ...
    [n] last cursor
    """
    import psycopg2
    mydb = psycopg2.connect(None, db, user, passwd, host)
    cursors = [mydb]
    for cursor_nb in range(number_of_cursor):
        cursors.append(mydb.cursor())
#        cursors[cursor_nb+1].execute("SET search_path TO %s"%schema)
    return cursors

def retrieve_RefSeq(mrna_accession=None, prot_accession=None):
    """
    Retrieve cross-reference information from UCSC database using a
    RefSeq accession.
    
    Returns a list.
    [0] gene symbol
    [1] gene complete name
    [2] mRNA accession
    [3] protein accession
    """
    import mysql.connector
    mydb = mysql.connector.connect(
            host='genome-mysql.cse.ucsc.edu', user='genome', db='hgFixed')
    cursor = mydb.cursor(buffered=True)
    cursor.execute(
            'SELECT mrnaAcc, protAcc, name, product FROM refLink WHERE mrnaAcc'\
                    ' = "%s" OR protAcc = "%s"'%(mrna_accession, prot_accession))
    if mrna_accession == None and prot_accession == None:
        return [None,None,None,None]
    else:
        return list(cursor.fetchone())

def retrieve_xref_Ensembl(stable_id=None,mrnaAcc=None,
        gene_acronym=None):
    """
    Retrieve cross-reference information from Ensembl.
    
    Returns a list.
    [0] gene stable id
    [1] transcript stable id
    [2] accession
    [3] gene id
    [4] transcript id
    [5] gene acronym (HGNC)
    [6] gene description
    """
    import mysql.connector
    mydb = mysql.connector.connect(
            host='ensembldb.ensembl.org', user='anonymous',
            db='homo_sapiens_core_86_38')
    cursor = mydb.cursor(buffered=True)
    base_sql = 'SELECT gene.stable_id, '\
            'transcript.stable_id, '\
            'xref.dbprimary_acc, '\
            'gene.gene_id, '\
            'transcript.transcript_id, '\
            'CASE WHEN gene_attrib.attrib_type_id = 4 '\
            'THEN gene_attrib.value '\
            'ELSE NULL END '\
            'AS gene_symbol, gene.description '\
            'FROM xref JOIN object_xref '\
            'ON xref.xref_id = object_xref.xref_id '\
            'JOIN transcript '\
            'ON transcript.transcript_id = object_xref.ensembl_id '\
            'JOIN gene '\
            'ON transcript.gene_id = gene.gene_id '\
            'JOIN gene_attrib '\
            'ON gene_attrib.gene_id = transcript.gene_id '\
            'WHERE '
    try:
        cursor.execute(base_sql+'(dbprimary_acc LIKE "N_\_%%" '\
                'OR dbprimary_acc LIKE "X_\_%%") '\
                'AND (gene.stable_id = "%s" '\
                'OR transcript.stable_id = "%s") '\
                'ORDER BY attrib_type_id'%(stable_id,stable_id))
        return list(cursor.fetchone())
    except:
        try:
            cursor.execute(base_sql+'dbprimary_acc = "%s" '\
                    'ORDER BY attrib_type_id'%mrnaAcc)
            return list(cursor.fetchone())
        except:
            cursor.execute(base_sql+'(dbprimary_acc LIKE "N_\_%%" '\
                    'OR dbprimary_acc LIKE "X_\_%%") '\
                    'AND value = "%s"'%gene_acronym)
            return list(cursor.fetchone())

def format_description(fas_description, verbose=False):
    '''
    Takes a fasta description line and try to retrieve informations out of it
    if it has a known format.
    
    Returns a dictionnary of available informations
    '''
    infos = {}
    try:
        infos = regex.match("(?<description>"\
                "(?<genome_assembly>\D\D\d+)"\
                "?(?:_"\
                "(?<source>[^_]*)"\
                ")?_?(?:"\
                "(?P<mrnaAcc>[N|X][M|R]_\d+)"\
                "|"\
                "(?P<protAcc>[N|X]P_\d+)"\
                ")(?:\.\d+)?(?: (range=)?(?<range>"\
                "(?<chromosome>chr.*):(?<start>(\d*))-(?<end>(\d*)))"\
                ")?(?: 5'pad=(?<pad5>\d*))?(?: 3'pad=(?<pad3>\d*))?"\
                "(?: strand=(?<strand>.))?"\
                "(?: repeatMasking=(?<repeatMasking>.*))?)",
                fas_description).groupdict()
    except:
        verbosify(verbose,"RefSeq not recognised for %s"%fas_description)
    try:
        infos = regex.match("(?<description>"\
                "(?:(?<identifier>[^ ]*)? )?"\
                "(?:((?<stable_id>ENS[T|G]\d*)|(?<chromosomal>dna:chromosome))"\
                "(?:\.\d)? )"\
                "(?:(?<info>[^ ]*) )?"\
                "(?:(chromosome:)?"\
                "(?:(?<genome_assembly>[^:]*):)?"\
                "(?<range>(?:(?<chromosome>[^:]*):)(?<start>\d*):(?<end>\d*)):"\
                "(?<strand>[+-1])(?:.*))?"\
                "(?:\|(?<exon_start>\d*)\|(?<exon_end>\d*))?(?:.*))",
                fas_description).groupdict()
        infos['source'] = "Ensembl"
    except:
        verbosify(verbose,"Ensembl not recognised for %s"%fas_description)
    if 'description' not in infos.keys() or infos.get('description') == '':
        try:
            try:
                infos = regex.search("(?<description>"\
                        "(?:.*)(GRCh\d\d:)?(hg\d\d:)?"\
                        "(?:(?<chromosome>(chr)?[^:]*):)?"\
                        "(?:(?<start>\d*)[:-](?<end>\d*):?)?"\
                        "(?<strand>[+-1])?"\
                        "(?:\|(?<exon_start>\d*)\|(?<exon_end>\d*)))",
                        fas_description).groupdict()
            except:
                try:
                    infos = regex.search("(?<description>"\
                            "(?:.*)"\
                            "(?: range=(?<range>"\
                            "(?<chromosome>chr.*):"\
                            "(?<start>(\d*))-(?<end>(\d*))))?"\
                            "(?: .*strand=(?<strand>.))?"\
                            "(?:\|(?<exon_start>\d*)\|(?<exon_end>\d*)))",
                            fas_description).groupdict()
                except:
                    infos = regex.search("(?<description>"\
                            "(?:.*)(GRCh\d\d:)?(hg\d\d:)?"\
                            "(?:(?<chromosome>(chr)?[^:]*):)?"\
                            "(?:(?<start>\d*)[:-](?<end>\d*):?)?"\
                            "(?<strand>[+-1]?)?)",
                            fas_description).groupdict()
        except:
            infos['description'] = fas_description
    if infos.get('start') == None and infos.get('exon_start'):
        infos['start'] = infos['exon_start']
    if infos.get('end') == None and infos.get('exon_end'):
        infos['end'] = infos['exon_end']
    if 'strand' in infos.keys() and infos.get('strand') == '1':
        infos['strand'] = '+'
    if infos.get('strand') not in ['+','-'] and infos.get('start')\
    and infos.get('end'):
        if int(infos.get('start')) <= int(infos.get('end')):
            infos['strand'] = '+'
        elif int(infos.get('start')) > int(infos.get('end')):
            infos['strand'] = '-'
            [infos['start'], infos['end']] = [infos.get('end'),infos.get('start')]
    if 'chromosome' in infos.keys() and infos.get('chromosome')\
    and infos.get('chromosome')[:3] != 'chr':
        infos['chromosome'] = 'chr' + infos['chromosome']
    if 'range' in infos.keys() and infos.get('range')\
    and infos.get('range')[:3] != 'chr':
        infos['range'] = 'chr' + infos['range']
    return infos

def fasta_fetcher(
        fasta_file,
        number_to_fetch,
        seq_size,
        verbose=False):
    """
    Fetch for sequences from a fasta file and returns a defined number of 
    random sequences or random window from random sequences if seq_size is
    not 0.
    
    number_to_fetch = 0 takes everything
    seq_size = 0 takes full length sequences
    
    Return a dictionnary.
    {Description:sequence}
    """
    from Bio import SeqIO
    fas_dic = OrderedDict()
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        if len(seq.seq) > seq_size and seq_size != 0:
            r_int = np.random.randint(0, len(seq.seq)-seq_size)
            fas_dic[seq.description] = str(seq.seq)[r_int:r_int+seq_size]
        else:
            fas_dic[seq.description] = str(seq.seq)
    dic = {}
    verbosify(verbose, "File fetched")
    if number_to_fetch == 0:
        return fas_dic
    else:
        randomize = np.random.permutation(len(fas_dic))
        for i in range(number_to_fetch):
            dic[fas_dic.keys()[randomize[i]]] = fas_dic[
                    fas_dic.keys()[randomize[i]]].strip('N').strip('n')
        return dic

def fasta_str_fetcher(fasta_string, verbose=False):
    """
    Fetch for sequences in a fasta file presented as a string.
    
    Return a dictionnary.
    {Description:sequence}
    """
    fas_dic = OrderedDict()
    for instance in regex.split(r'\\r\\n>|\\n>|>', fasta_string)[1:]:
        [description, seq] = regex.split(r'\\r\\n|\\n', instance, maxsplit=1)
        fas_dic[description] = regex.sub(r'\\r\\n|\\n','', seq)
    return fas_dic

def kmer_transfo(
        df_,
        depth,
        sort_column,
        sequence_column,
        target_column,
        window,
        jellyfish=False,
        overlapped=True,
        verbose=False):
    """
    Define sequences by their kmers proportions and returns a bigger
    dataframe containing it.
    
    jellyfish = True uses jellyfish command to get the kmers
    overlapped = True allows kmers to overlapped each other (almost
    always the case)
    
    Return pandas dataframe.
    """
    df = df_.copy()
    nts = ['A','U','C','G']
    di_nts = []
    if depth == 1:
        di_nts = nts
    else:
        for nt1 in nts:
            for nt2 in nts:
                if depth == 2:
                    di_nts.append(nt1+nt2)
                elif depth == 3:
                    for nt3 in nts:
                        di_nts.append(nt1+nt2+nt3)
                elif depth == 4:
                    for nt3 in nts:
                        for nt4 in nts:
                            di_nts.append(nt1+nt2+nt3+nt4)
                else:
                    print "This kmer length isn't available"
                    break
    for each in di_nts:
        df[each] = .0
    lst_rows = []
    pos_dframe = pd.DataFrame(columns=df.columns)
    for row in df.index:
        lst_rows.append(row)
        if jellyfish is True:
            di_nt_cnts = {}
            for line_out in subprocess.check_output(u'echo ">0\n%s" | "\
                    "sed "s/U/T/g" | "\
                    "jellyfish count -m 3 -s 100 -o /dev/stdout /dev/stdin | "\
                    "jellyfish dump -ct /dev/stdin | "\
                    "sed "s/T/U/g"'%(df.loc[row,sequence_column].
    upper().replace('T','U')), shell=True).split('\n')[:-1]:
                di_nt_cnts[line_out.split('\t')[0]] = int(line_out.split('\t')[1])
        else:
            di_nt_lst = regex.findall(
                    '.{%d}'%depth,df.loc[row,sequence_column].upper().replace('T','U'),
                    overlapped=True)
            di_nt_cnts = Counter(di_nt_lst)
        if len([di_nt_cnts[x] for x in di_nt_cnts]) > 4**depth:
            print "Crap! There's a non AUCG nt in the sequence at row %d"%row
            break
        total_di_nt = sum(di_nt_cnts.values())
        di_nt_freqs = [(str(di_nt), float(di_nt_cnts[di_nt])/total_di_nt)
                for di_nt in di_nt_cnts if "N" not in di_nt]
        for di_ntd, freq in di_nt_freqs:
            df.loc[row,di_ntd] = freq
    verbosify(verbose, "Kmer transformed")
    return df

def trimer_transfo(
        df_,
        sequence_column,
        verbose=False):
    """
    Define sequences by their 3mers proportions and returns a bigger
    dataframe containing it.
    This version always considers overlapping trimers.
    
    Return pandas dataframe.
    """
    df = df_.copy()
    nts = ['A','U','C','G']
    tri_nts = []
    for nt1 in nts:
        for nt2 in nts:
            for nt3 in nts:
                tri_nts.append([nt1+nt2+nt3,
                    "(?P<"+nt1+nt2+nt3+">"+nt1+"(?="+nt2+nt3+"))"])
    for each, pattern in tri_nts:
        df[each] = df[sequence_column].str.upper().str.replace(
                'T','U').str.count(pattern)/(df[sequence_column].str.len()-2)
    verbosify(verbose, "trimer transformed")
    return df
