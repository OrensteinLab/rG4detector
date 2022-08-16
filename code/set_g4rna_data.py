import pandas as pd
import re
from Bio import pairwise2
import os
from PARAMETERS import *
import sys

GAPS = DATA_SIZE
MATCH_THRESHOLD = 0.85


def get_chrom_data(chrom_data):
    chrom_data.sort_values("Start", inplace=True)
    chrom = chrom_data.iloc[0]["Chromosome"]
    strand = chrom_data.iloc[0]["Strand"]

    i = 0
    data = {"chromosome": [], "start": [], "length": [], "strand": [], "label": [], "sequence": []}
    while i < len(chrom_data):
        valid = True
        start = chrom_data.iloc[i]["Start"]
        length = chrom_data.iloc[i]["Length"]
        label = chrom_data.iloc[i]["G4"]
        seq = chrom_data.iloc[i]["Sequence"].upper().replace("U", "T")
        i += 1
        while i < len(chrom_data) and chrom_data.iloc[i]['Start'] <= start + length:
            if label != chrom_data.iloc[i]["G4"]:
                valid = False
            # choose contained seq
            if chrom_data.iloc[i]['Start'] + chrom_data.iloc[i]['Length'] <= start + length:
                start = chrom_data.iloc[i]["Start"]
                length = chrom_data.iloc[i]["Length"]
                seq = chrom_data.iloc[i]["Sequence"].upper().replace("U", "T")
            else:
                length = chrom_data.iloc[i]['Length'] + chrom_data.iloc[i]["Start"] - start
            i += 1
        if valid:
            data["start"].append(start)
            data["length"].append(length)
            data["label"].append(label)
            data["strand"].append(strand)
            data["chromosome"].append(chrom)
            data["sequence"].append(seq)

    return pd.DataFrame(data)


def filter_data(input_file, csv_dest):
    data = pd.read_csv(input_file, delimiter="\t")
    chrom_list = []
    for i in range(1, 23):
        chrom_list.append(f'chr{i}')
    chrom_list.append('chrX')
    chrom_list.append('chrY')

    all_chrom_data = []
    for chrom in chrom_list:
        for strand in ["+", "-"]:
            chrom_data = data[(data['Chromosome'] == chrom) & (data['Strand'] == strand)].copy()
            if len(chrom_data) > 0:
                all_chrom_data.append(get_chrom_data(chrom_data))
    all_data = pd.concat(all_chrom_data)
    all_data.reset_index(drop=True, inplace=True)
    all_data.to_csv(csv_dest)


def csv2bed(csv_path, bed_dest):
    bed_file = open(bed_dest, 'w')
    interval = pd.read_csv(csv_path)
    for _, row in interval.iterrows():
        chrom = row['chromosome']
        length = row['length']
        start = row['start']
        strand = row['strand']
        end = start + length + GAPS
        start = start - GAPS
        bed_file.write(f'{chrom}\t{int(start)}\t{int(end)}\tname\t0\t{strand}\n')
    bed_file.close()


def find_g4rna_seq(raw_csv_path, seq_dest, csv_dest):
    no_hit_counter = 0
    hit_counter = 0
    ignore_list = []
    seq_file = open(seq_dest + ".txt")
    seq_list = [s.upper() for s in seq_file.read().splitlines()]
    g4rna_df = pd.read_csv(raw_csv_path)
    g4rna_list = g4rna_df["sequence"].to_list()
    for idx, (seq, g4rna_seq) in enumerate(zip(seq_list, g4rna_list)):
        seq_match = re.search(g4rna_seq, seq)
        if seq_match is None:
            alignments = pairwise2.align.localms(g4rna_seq, seq, 1, -.8, -.8, -.8)
            if len(alignments) > 0 and alignments[0].score / len(g4rna_seq) > MATCH_THRESHOLD:
                hit_counter += 1
            else:
                # print(g4rna_df.iloc[idx]["sequence"])
                no_hit_counter += 1
                ignore_list.append(idx)
                # print(g4rna_seq)
        else:
            hit_counter += 1
    print(f"HIT = {hit_counter}")
    print(f"NO-HIT = {no_hit_counter}")
    g4rna_df.drop(ignore_list, inplace=True)
    g4rna_df.to_csv(csv_dest, index=False)


def main(g4rna_dir, reference_genome):
    source = g4rna_dir + "/data.csv"
    raw_csv_dest = g4rna_dir + "/g4rna_filtered_data_raw.csv"
    csv_dest = g4rna_dir + "/g4rna_filtered_data.csv"
    raw_bed_dest = g4rna_dir + "/bed_raw_data.bed"
    bed_dest = g4rna_dir + "/bed_raw_data.bed"
    seq_dest = g4rna_dir + "/seq"
    raw_seq_dest = g4rna_dir + "/raw_seq"
    bedtools_script_path = g4rna_dir + "/bed2seq.sh"

    # filter overlapping sequences
    filter_data(source, raw_csv_dest)
    csv2bed(raw_csv_dest, raw_bed_dest)
    # crete raw sequences files
    os.system(f"bash {bedtools_script_path} {reference_genome}  {raw_bed_dest} {raw_seq_dest}")
    # take only sequences that match the genome coordinate
    find_g4rna_seq(raw_csv_dest, raw_seq_dest, csv_dest)
    # crate update bed file
    csv2bed(csv_dest, bed_dest)
    # crete sequences files
    os.system(f"bash {bedtools_script_path} {reference_genome} {bed_dest} {seq_dest}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: set_g4rna_data.py <g4rna_dir_path> <hg38.fa>")
        exit(0)
    g4rna_dir_path = sys.argv[1]
    reference_genome = sys.argv[2]
    main(g4rna_dir_path, reference_genome)

