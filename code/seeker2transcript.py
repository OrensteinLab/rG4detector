from Bio import SeqIO
import re
import pandas as pd
import numpy as np
import time
import pickle
from Bio import pairwise2
from PARAMETERS import *
from math import isnan

MATCH_THRESHOLD = 0.9

def get_seeker_df(data_path):
    diagram_col = "Sequence diagram (RTS sites are indicated by asterisks)"
    gene_col = "Overlapping Gencode \ngene names"
    interval_col = "Genomic intervals of rG4"

    rg4_seq_df = pd.read_csv(data_path)
    for idx, row in rg4_seq_df.iterrows():
        if not isinstance(row[gene_col], str):  # ignore NaNs
            continue
        start = []
        end = []
        rG4_intervals = row[interval_col].split(',')
        for interval in rG4_intervals:
            rng = interval.split(':')[1]
            start.append(int(rng.split('-')[0]))
            end.append(int(rng.split('-')[1]))
        length = 0
        for s, e in zip(start, end):
            length += e - s
        seq = re.findall("5'- ([GTCA]+)", row[diagram_col])
        if len(seq) != 1:
            print(f"ERROR in row {idx}, seq = {seq}")
            exit()
        gene = re.findall("ENSG\d*\.\d+", row[gene_col])[0]
        rg4_seq_df.at[idx, "sequence"] = seq[0][:length]
        rg4_seq_df.at[idx, "gene"] = gene
    return rg4_seq_df.loc[:, ["gene", "sequence"]]


def get_transcript_dict(fasta_path):
    transcript_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    keys_tuple_list = []
    for k in transcript_dict.keys():
        new_key = re.findall("ENST\d*\.\d+", k)
        if len(new_key) != 1:
            print(f"found {len(new_key)} matchs in {k}")
        else:
            keys_tuple_list.append((k, new_key))
    for k, new_key in keys_tuple_list:
        transcript_dict[new_key[0]] = transcript_dict[k]
        del transcript_dict[k]
    return transcript_dict


def set_transcript_match_dict(rg4_seq_df, transcript_dict):
    transcript_match_dict = {}
    for idx, row in rg4_seq_df.iterrows():
        seq = row["sequence"]
        for trans_name in row["transcripts"]:
            if trans_name not in transcript_dict.keys():
                continue
            seq_match = re.search(seq, str(transcript_dict[trans_name].seq))
            if seq_match is None:
                continue
            seq_location = seq_match.span()
            location_arr = np.zeros(len(transcript_dict[trans_name].seq), dtype=bool)
            location_arr[seq_location[0]:seq_location[1]] = True
            if trans_name in transcript_match_dict:
                transcript_match_dict[trans_name] = transcript_match_dict[trans_name] | location_arr
            else:
                transcript_match_dict[trans_name] = location_arr
            row["transcripts"].remove(trans_name)
            break

    # search for transcripts with multiple sequences
    for idx, row in rg4_seq_df.iterrows():
        seq = row["sequence"]
        for trans_name in row["transcripts"]:
            if trans_name not in transcript_match_dict:
                continue
            seq_match = re.search(seq, str(transcript_dict[trans_name].seq))
            if seq_match is None:
                alignments = pairwise2.align.localms(seq, str(transcript_dict[trans_name].seq), 1, -.8, -.8, -.8)
                if alignments[0].score / len(seq) > MATCH_THRESHOLD:
                    seq_location = (alignments[0].start, alignments[0].end)
                else:
                    continue
            else:
                seq_location = seq_match.span()
            location_arr = np.zeros(len(transcript_dict[trans_name].seq), dtype=bool)
            location_arr[seq_location[0]:seq_location[1]] = True
            transcript_match_dict[trans_name] = transcript_match_dict[trans_name] | location_arr

    print(f"total transcripts = {len(transcript_match_dict)}")
    return transcript_match_dict


def print_seeker_fasta(transcript_match_dict, transcript_dict):
    temp_dict = {}
    for trans_name in transcript_match_dict:
        temp_dict[trans_name] = transcript_dict[trans_name]
        temp_dict[trans_name].id = trans_name
        temp_dict[trans_name].description = ''
    with open(SCREENER_DETECTION_PATH + "seeker-test.fasta", "w") as f:
        SeqIO.write(temp_dict.values(), f, "fasta")


def main():
    t = time.time()
    rg4_seq_df = get_seeker_df(RG4_SEEKER_PATH)
    # transcript_dict = get_transcript_dict(HUMAN_TRANSCRIPTOME_PATH)
    test_df = rg4_seq_df[rg4_seq_df["chromosome"] == "chr1"]
    output_file = DETECTION_RG4_SEEKER_HITS
    transcript_match_dict = set_transcript_match_dict(test_df, transcript_dict)
    with open(output_file, "wb") as fp:  # Pickling
        pickle.dump(transcript_match_dict, fp)
    print_seeker_fasta(transcript_match_dict, transcript_dict)
    print(f"Execution time: {round(time.time() - t)}s")


if __name__ == "__main__":
    main()

