from Bio import SeqIO
import pandas as pd
import numpy as np
import pickle
from PARAMETERS import *


def create_mouse_fasta():
    data_file = pd.read_csv(MOUSE_DATA_PATH)
    records_dict = SeqIO.to_dict(SeqIO.parse(MOUSE_TRANSCRIPTOME_PATH, "fasta"))
    file_out = open(MOUSE_PATH + "mouse.fa", "w")
    data_rts = []
    for _, row in data_file.iterrows():
        transcript = row['Transcript']
        # get rsr data
        rts_ratio = row["Untreated (K+)"] / row['Untreated (Li+)']
        if transcript in records_dict and np.isfinite(rts_ratio):
            # get seq
            pos = row['RT-stop position']
            seq = records_dict[transcript].seq
            start = max(0, pos - 30 - (MOUSE_SCREENER_LENGTH - 30) // 2)
            end = min(len(seq), pos + (MOUSE_SCREENER_LENGTH - 30) // 2)
            seq = str(seq[start:end])
            header = '>' + row['Transcript'] + ": " + str(start) + '-' + str(end)
            if len(seq) > 0:
                # write to fasta
                file_out.write(header + "\n" + seq + "\n")
                # save rts data
                data_rts.append(rts_ratio)
    with open(MOUSE_PATH + 'mouse_rts.pkl', 'wb') as fp:
        pickle.dump(data_rts, fp, protocol=pickle.HIGHEST_PROTOCOL)
    file_out.close()


if __name__ == "__main__":
    create_mouse_fasta()
