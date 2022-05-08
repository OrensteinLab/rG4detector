from Bio import SeqIO
import re
import pandas as pd
import sys
import numpy as np
import time
import pickle
from Bio import pairwise2


MATCH_THRESHOLD = 0.9
TEST_SET = False
PDS = False
output = "detection/"
# transcripts_file = f"big_files/gencode.v29.transcripts.fa"
transcripts_file = "big_files/gencode.v29lift37.transcripts.fa"
seeker_path = "rg4-seq_rts.k.rG4_list.combined.csv"
# seeker_path = "rG4-seeker-hits.csv"
if PDS is True:
    seeker_path = "rg4-seq_rts.kpds.rG4_list.combined.csv"
    output = "kpds/" + output

if len(sys.argv) > 1:
    transcripts_file = sys.argv[1]
if len(sys.argv) > 2:
    output = sys.argv[2]


def get_seeker_df(data_path):
    rg4_seq_df = pd.read_csv(data_path, usecols=["sequence_diagram", "'GENCODE_V29'_coding_regions", "rG4_intervals",
                                                 "chromosome", "rG4_structural_class", "splicing"])
    rg4_seq_df.rename(columns={"'GENCODE_V29'_coding_regions": "transcripts", "sequence_diagram": "sequence",
                               "rG4_structural_class": "type"},
                      inplace=True)
    rg4_seq_df = rg4_seq_df[rg4_seq_df["splicing"] == 0].drop("splicing", axis=1).reset_index(drop=True)
    length = []
    for idx, row in rg4_seq_df.iterrows():
        rG4_intervals = row['rG4_intervals'].split(':')
        start = int(rG4_intervals[1].split('-')[0])
        end = int(rG4_intervals[1].split('-')[1])
        seq = re.findall("5'- ([GTCA]+)", row["sequence"])
        if len(seq) != 1:
            print(f"ERROR in row {idx}, seq = {seq}")
            exit()
        transcripts = row["transcripts"].split(";")
        for t in range(len(transcripts)):
            transcripts[t] = re.findall("ENST\d*\.\d+", transcripts[t])[0]
        rg4_seq_df.at[idx, "sequence"] = seq[0]  # [:end-start]
        rg4_seq_df.at[idx, "transcripts"] = transcripts
        length.append(end-start)
    print(sum(length)/len(length))
    return rg4_seq_df.drop("rG4_intervals", axis=1)

def get_seeker_df_ver2(data_path):
    rg4_seq_df = pd.read_csv(data_path, usecols=["sequence_diagram", "genes", "rG4_intervals",
                                                 "chromosome", "rG4_structural_class", "splicing"])
    rg4_seq_df.rename(columns={"sequence_diagram": "sequence",
                               "rG4_structural_class": "type"},
                      inplace=True)
    rg4_seq_df.dropna(subset=["genes"], inplace=True)
    rg4_seq_df = rg4_seq_df[rg4_seq_df["splicing"] == 0].drop("splicing", axis=1).reset_index(drop=True)
    for idx, row in rg4_seq_df.iterrows():
        rG4_intervals = row['rG4_intervals'].split(':')
        start = int(rG4_intervals[1].split('-')[0])
        end = int(rG4_intervals[1].split('-')[1])
        seq = re.findall("5'- ([GTCA]+)", row["sequence"])
        if len(seq) != 1:
            print(f"ERROR in row {idx}, seq = {seq}")
            exit()
        gene_names = re.findall("ENSG\d+\.\d+", row["genes"])
        # gene_names = row["genes"].split("|")[1:]
        rg4_seq_df.at[idx, "sequence"] = seq[0][:end-start]
        rg4_seq_df.at[idx, "genes"] = gene_names
    return rg4_seq_df.drop("rG4_intervals", axis=1)


def get_transcript_dict(fasta_path, by_gene=False):
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


def get_transcript_dict_ver2(fasta_path):
    transcript_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    output_dict = {}
    for k in transcript_dict.keys():
        gene_key = re.findall("ENSG\d*\.\d+", k)
        trans_key = re.findall("ENST\d*\.\d+", k)
        if len(gene_key) != 1 or len(trans_key) != 1:
            print(f"found {len(max(gene_key, trans_key))} matchs in {k}")
        else:
            if gene_key[0] in output_dict:
                output_dict[gene_key[0]][trans_key[0]] = transcript_dict[k]
            else:
                output_dict[gene_key[0]] = {trans_key[0]: transcript_dict[k]}
    return output_dict


def set_trans_names(rg4_df, transcript_dict):
    new_trans_dict = {}
    trans_col = []
    miss_counter = 0
    for idx, row in rg4_df.iterrows():
        genes = row["genes"]
        seq = row["sequence"]
        trans_list = []
        for g in genes:
            if g in transcript_dict.keys():
                for t in transcript_dict[g].keys():
                    res = transcript_dict[g][t].seq.find(seq)
                    if res != -1:
                        trans_list.append(t)
                        new_trans_dict[t] = transcript_dict[g][t]
        # if len(trans_list) == 0:  # align seq to find similar sub-seq
        #     for g in genes:
        #         if g in transcript_dict.keys():
        #             for t in transcript_dict[g].keys():
        #                 alignments = pairwise2.align.localms(seq, str(transcript_dict[g][t].seq), 1, -.8, -.8, -.8)
        #                 # print(alignments[0].score / len(seq))
        #                 if alignments[0].score / len(seq) > MATCH_THRESHOLD:
        #                     trans_list.append(t)

        if len(trans_list) == 0:
            # print(f"No mach for {idx}")
            rg4_df.drop([idx], inplace=True)
            miss_counter += 1
        else:
            trans_col.append(trans_list)
            # rg4_df.loc[idx, "transcripts"] = trans_list
    print(f"miss counter = {miss_counter}")
    rg4_df["transcripts"] = trans_col
    return rg4_df, new_trans_dict





def get_seq_positions(annotation_file, chr_start, seq_length):
    end = None
    offset = 0

    if chr_start < annotation_file.at[1, "start"]:
        end = chr_start - annotation_file.at[1, "start"] + seq_length
        if end < 0:
            offset = None
            print("ERROR: get_seq_positions: end < 0")
    else:
        for idx, row in annotation_file.iterrows():
            if row["end"] < chr_start:
                offset += row["end"] - row["start"] + 1
            else:
                if row["start"] > chr_start:
                    offset = None
                    end = 0
                else:
                    offset = offset + chr_start - row["start"] + 1
                    end = offset + seq_length
                break

    if end is None:
        print("ERROR: get_seq_positions: end is None")
        exit(1)
    return offset, end


def set_transcript_match_dict(rg4_seq_df, transcript_dict):
    transcript_match_dict = {}
    exist_counter = 0
    missing_trans = 0
    no_match_counter = 0
    for idx, row in rg4_seq_df.iterrows():
        seq = row["sequence"]
        no_trans = True
        no_match = False
        seq_exist = False
        for trans_name in row["transcripts"]:
            if trans_name not in transcript_dict.keys():
                continue
            seq_match = re.search(seq, str(transcript_dict[trans_name].seq))
            if seq_match is None:
                no_trans = False
                no_match = True
                continue
            no_trans = False
            no_match = False
            seq_exist = True
            seq_location = seq_match.span()
            location_arr = np.zeros(len(transcript_dict[trans_name].seq), dtype=bool)
            location_arr[seq_location[0]:seq_location[1]] = True
            if trans_name in transcript_match_dict:
                transcript_match_dict[trans_name] = transcript_match_dict[trans_name] | location_arr
            else:
                transcript_match_dict[trans_name] = location_arr
            row["transcripts"].remove(trans_name)
            break

        if seq_exist:
            exist_counter += 1
        elif no_match:
            no_match_counter += 1
        elif no_trans:
            missing_trans += 1
        else:
            print("ERROR: no label to seq")

    # search for transcripts with multiple sequences
    no_match_2 = 0
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
                    no_match_2 += 1
                    continue
            else:
                seq_location = seq_match.span()
            location_arr = np.zeros(len(transcript_dict[trans_name].seq), dtype=bool)
            location_arr[seq_location[0]:seq_location[1]] = True
            transcript_match_dict[trans_name] = transcript_match_dict[trans_name] | location_arr
            exist_counter += 1

    print(f"missing_counter = {no_match_counter}")
    print(f"no_match_2 = {no_match_2}")
    print(f"missing_trans = {missing_trans}")
    print(f"exist_counter = {exist_counter}")
    print(f"total transcripts = {len(transcript_match_dict)}")
    return transcript_match_dict


def print_seeker_fasta(transcript_match_dict, transcript_dict, path="detection/screener_detection/seeker-test.fasta"):
    if PDS is True:
        path = "kpds/detection/screener/seeker.fasta"
    temp_dict = {}
    for trans_name in transcript_match_dict:
        temp_dict[trans_name] = transcript_dict[trans_name]
        temp_dict[trans_name].id = trans_name
        temp_dict[trans_name].description = ''
    with open(path, "w") as f:
        SeqIO.write(temp_dict.values(), f, "fasta")


def main():
    t = time.time()
    new_dict = True
    rg4_seq_df = get_seeker_df(seeker_path)
    transcript_dict = get_transcript_dict(transcripts_file)
    # rg4_seq_df, transcript_dict = set_trans_names(rg4_seq_df, transcript_dict)
    if new_dict:
        sets_dict = {"test": rg4_seq_df[rg4_seq_df["chromosome"] == "chr1"],
                     "val": rg4_seq_df[rg4_seq_df["chromosome"] == "chr2"],
                     "train": rg4_seq_df[(rg4_seq_df["chromosome"] != "chr1") & (rg4_seq_df["chromosome"] != "chr2")]}
        for data_set in ["test", "train", "val"]:
            output_file = output + f"rg4-seeker-transcript-match-{data_set}.pkl"
            transcript_match_dict = set_transcript_match_dict(sets_dict[data_set], transcript_dict)
            with open(output_file, "wb") as fp:  # Pickling
                pickle.dump(transcript_match_dict, fp)
    # create fasta for screener
    with open(output + "rg4-seeker-transcript-match-test.pkl", 'rb') as fp:
        transcript_match_dict = pickle.load(fp)
    print_seeker_fasta(transcript_match_dict, transcript_dict)

    # save data
    print(time.time() - t)


if __name__ == "__main__":
    main()

