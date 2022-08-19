import pandas as pd
import re
from Bio import pairwise2, SeqIO, Seq
import os
from PARAMETERS import *
import sys
from time import time


GAP3 = (DATA_SIZE - 30)//2
GAP5 = DATA_SIZE - GAP3 - 11
MATCH_THRESHOLD = 0.85


def set_hg38_locations(data, data_loc, ref_gen):
    extra = 10
    no_hit_counter = 0
    hit_counter = 0
    ignore_list = []
    t = time()
    align = 0
    search = 0
    # ref_gen = "../../rg4detector_dev/big_files/chr1.fa"
    data_df = pd.read_csv(data, delimiter="\t").dropna()
    genome_dict = SeqIO.to_dict(SeqIO.parse(ref_gen, "fasta"))
    for idx, row in data_df.iterrows():
        if (idx+1) % 50 == 0:
            print(f"Done {idx+1}/{len(data_df)}, {round(time()-t)} seconds")
            t = time()
        start = int(row["Start"]) - extra
        end = int(row["Length"]) + int(row["Start"]) + extra
        chrom = row["Chromosome"]
        seq = row["Sequence"].upper().replace("U", "T")
        # if chrom != "chr1":
        #     continue
        rg4_seq_range = genome_dict[chrom].seq[start:end]

        if row["Strand"] == "-":
            rg4_seq_range = rg4_seq_range.reverse_complement()


        seq_match = re.search(seq, str(rg4_seq_range))
        if seq_match:
            data_df.at[idx, "Start"] = start + seq_match.span()[0]
            hit_counter += 1
            continue

        if len(seq) < 200:
            alignments = pairwise2.align.localms(seq, str(rg4_seq_range), 1, -.8, -.8, -.8)
            if len(alignments) > 0 and alignments[0].score / len(seq) > MATCH_THRESHOLD:
                data_df.at[idx, "Start"] = alignments[0].start + start
                align += 1
                continue

        no_hit_counter += 1
        ignore_list.append(idx)

    print(f"search={search}, align={align}")
    print(f"HIT = {hit_counter}")
    print(f"NO-HIT = {no_hit_counter}")
    data_df.drop(ignore_list, inplace=True)
    data_df.to_csv(data_loc, index=False)


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
                break
            # choose containing seq
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
    data = pd.read_csv(input_file)#, delimiter="\t")
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
    print(f"Number of sequences = {len(all_data)}")
    all_data.to_csv(csv_dest)


def csv2bed(csv_path, bed_dest, gap3=GAP3, gap5=GAP5):
    bed_file = open(bed_dest, 'w')
    interval = pd.read_csv(csv_path)
    for _, row in interval.iterrows():
        chrom = row['chromosome']
        length = row['length']
        strand = row['strand']

        if strand == "+":
            start = row['start'] - gap5
            end = row['start'] + length + gap3
        else:
            start = row['start'] - gap3
            end = row['start'] + length + max(gap5, DATA_SIZE-gap3-length)

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
                no_hit_counter += 1
                ignore_list.append(idx)
        else:
            hit_counter += 1


    print(f"HIT = {hit_counter}")
    print(f"NO-HIT = {no_hit_counter}")
    g4rna_df.drop(ignore_list, inplace=True)
    g4rna_df.to_csv(csv_dest, index=False)


def main(g4rna_dir, ref_gen):
    source = g4rna_dir + "/data.csv"
    data_loc = g4rna_dir + "/data_loc.csv"
    raw_csv_dest = g4rna_dir + "/g4rna_filtered_data_raw.csv"
    csv_dest = g4rna_dir + "/g4rna_filtered_data.csv"
    raw_bed_dest = g4rna_dir + "/bed_raw_data.bed"
    bed_dest = g4rna_dir + "/bed_raw_data.bed"
    bed_dest_no_gap = g4rna_dir + "/bed_raw_data_no_gap.bed"
    seq_dest = g4rna_dir + "/seq"
    seq_dest_no_gap = g4rna_dir + "/seq_no_gap"
    raw_seq_dest = g4rna_dir + "/raw_seq"
    bedtools_script_path = g4rna_dir + "/bed2seq.sh"

    # set correct locations
    # set_hg38_locations(source, data_loc, ref_gen)
    # filter overlapping sequences
    filter_data(data_loc, csv_dest)
    csv2bed(csv_dest, bed_dest)
    # crete raw sequences files
    os.system(f"bash {bedtools_script_path} {ref_gen} {bed_dest} {seq_dest}")
    csv2bed(csv_dest, bed_dest, gap3=0, gap5=0)
    os.system(f"bash {bedtools_script_path} {ref_gen} {bed_dest_no_gap} {seq_dest_no_gap}")

    # take only sequences that match the genome coordinate
    # find_g4rna_seq(raw_csv_dest, raw_seq_dest, csv_dest)
    # # crate update bed file
    # csv2bed(csv_dest, bed_dest)
    # # crate update bed file no gaps
    # csv2bed(csv_dest, bed_dest_no_gap, gap3=0, gap5=0)
    # # crete sequences files
    # os.system(f"bash {bedtools_script_path} {ref_gen} {bed_dest} {seq_dest}")
    # os.system(f"bash {bedtools_script_path} {ref_gen} {bed_dest_no_gap} {seq_dest_no_gap}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: set_g4rna_data.py <g4rna_dir_path> <hg38.fa>")
        exit(0)
    g4rna_dir_path = sys.argv[1]
    reference_genome = sys.argv[2]
    main(g4rna_dir_path, reference_genome)

