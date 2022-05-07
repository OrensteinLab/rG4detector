import time
import sys
import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_input_size, one_hot_enc, pred_all_sub_seq
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from sklearn.preprocessing import StandardScaler
from PARAMETERS import *


def predict_fasta(model, src, dst):
    with open(src) as f:
        f_lines = f.read().splitlines()
    seqs = f_lines[1::2]
    print(f"Number of sequences = {len(seqs)}")
    scores_df = pd.DataFrame(index=range(1, len(seqs)+1), dtype=float)
    for idx, seq in enumerate(seqs):
        if (idx+1) % 500 == 0:
            print(f"{idx+1} sequences are done")
        pred = make_prediction(model, seq)
        scores_df.loc[idx+1, "sequence"] = seq
        scores_df.loc[idx+1, "rG4detector"] = pred
    scores_df.to_csv(dst, index=False)


def make_prediction(model, seq, max_pred=True):
    one_hot_mat = one_hot_enc(seq, remove_last=False)
    preds = pred_all_sub_seq(one_hot_mat, model)
    if max_pred:
        return max(preds)
    else:
        return preds


if __name__ == "__main__":
    # fasta_file_path = sys.argv[1]
    fasta_file_path = "G3BP1/stress/G3BP1_2021_stress.fa"
    output = "predict_fasta_output.csv"

    if len(sys.argv) > 2:
        output = sys.argv[2]

    rG4detector_model = []
    for i in range(ENSEMBLE_SIZE):
        rG4detector_model.append(load_model(MODEL_PATH + f"model_{i}.h5"))
    predict_fasta(model=rG4detector_model, src=fasta_file_path, dst=output)



