from PARAMETERS import *
from Bio import SeqIO
import re
import pandas as pd
import sys
import numpy as np
import time
import pickle
from Bio import pairwise2


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
    return rg4_seq_df.loc[:, ["chromosome", "gene", "sequence", "strand"]].dropna().reset_index()



def get_transcript_dict(fasta_path, by_gene=False):
    transcript_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    keys_tuple_list = []
    for k in transcript_dict.keys():
        new_keys = re.findall("(ENS\w\d*\.\d+)", k)
        assert len(new_keys) == 2, f"ERROR didnt find tr & gene ID in {k}"
        keys_tuple_list.append((k, new_keys))

    if by_gene:
        for k, new_keys in keys_tuple_list:
            if new_keys[1] in transcript_dict:
                transcript_dict[new_keys[1]][new_keys[0]] = transcript_dict[k]
            else:
                transcript_dict[new_keys[1]] = {new_keys[0]: transcript_dict[k]}
            del transcript_dict[k]

    else:
        for k, new_keys in keys_tuple_list:
            transcript_dict[new_keys[0]] = transcript_dict[k]
            del transcript_dict[k]
    return transcript_dict


def sort_tr_by_tsl(tr_list):
    sorted_tr_list = []
    for tr in tr_list:
        pos = 0
        while pos < len(sorted_tr_list):
            if tr.tsl < sorted_tr_list[pos].tsl or \
                    (tr.tsl == sorted_tr_list[pos].tsl and tr.len > sorted_tr_list[pos].len):
                break
            pos += 1
        sorted_tr_list = sorted_tr_list[:pos] + [tr] + sorted_tr_list[pos:]
    return sorted_tr_list


def get_gene_tsl_info(info_path):
    new_dict = {}
    with open(info_path, 'rb') as f:
        chr_dict = pickle.load(f)
    for c in chr_dict:
        for s in chr_dict[c]:
            for tr in chr_dict[c][s]:
                if tr.gene in new_dict:
                    new_dict[tr.gene].append(tr)
                else:
                    new_dict[tr.gene] = [tr]
    del chr_dict
    return new_dict


def get_seq_pos(seq, tr):
    seq_match = re.search(seq, tr)
    if seq_match is None:
        return None
    seq_location = seq_match.span()
    location_arr = np.zeros(len(tr), dtype=bool)
    location_arr[seq_location[0]:seq_location[1]] = True
    return location_arr


def set_rG4_locations(rg4_seq_df, fasta_path):
    # get all transcripts dict
    transcript_dict = get_transcript_dict(fasta_path, by_gene=True)
    # get transcripts tsl info
    tsl_info = get_gene_tsl_info(tsl_info_path)

    # find rG4 locations in prominent transcripts
    missing_sequences = 0
    gene_rG4_locations = {}
    for idx, row in rg4_seq_df.iterrows():
        tr_list = sort_tr_by_tsl(tsl_info[row["gene"]])
        for i, tr in enumerate(tr_list):
            location_arr = get_seq_pos(row["sequence"], str(transcript_dict[tr.gene][tr.name].seq))
            if location_arr is None:
                if i == len(tr_list) - 1:
                    missing_sequences += 1
                continue
            if tr.gene in gene_rG4_locations:
                if tr.name in gene_rG4_locations[tr.gene]:
                    gene_rG4_locations[tr.gene][tr.name] = gene_rG4_locations[tr.gene][tr.name] | location_arr
                else:
                    gene_rG4_locations[tr.gene][tr.name] = location_arr
            else:
                gene_rG4_locations[tr.gene] = {tr.name: location_arr}
            rg4_seq_df.loc[idx, "transcript"] = tr.name
            break
    print(f"missing sequences = {missing_sequences}")

    # search for transcripts with multiple sequences
    transcripts_rG4_locations = {}
    for gene in gene_rG4_locations:
        gene_df = rg4_seq_df[rg4_seq_df["gene"] == gene]
        for tr in gene_rG4_locations[gene]:
            for idx, row in gene_df.iterrows():
                if row["transcript"] != tr:
                    location_arr = get_seq_pos(row["sequence"], str(transcript_dict[gene][tr].seq))
                    if location_arr is None:
                        continue
                    gene_rG4_locations[gene][tr] = gene_rG4_locations[gene][tr] | location_arr
            transcripts_rG4_locations[tr] = gene_rG4_locations[gene][tr]
    return transcripts_rG4_locations


def print_seeker_fasta(transcript_match_dict, transcript_dict_path):
    temp_dict = {}
    transcript_dict = get_transcript_dict(transcript_dict_path)
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
    transcript_match_dict = set_rG4_locations(test_df, HUMAN_V29_TRANSCRIPTOME_PATH)
    with open(DETECTION_RG4_SEEKER_HITS, "wb") as fp:  # Pickling
        pickle.dump(transcript_match_dict, fp)
    print_seeker_fasta(transcript_match_dict, HUMAN_V29_TRANSCRIPTOME_PATH)
    print(f"Execution time: {round(time.time() - t)}s")


if __name__ == "__main__":
    main()

