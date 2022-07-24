import time
import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_input_size, one_hot_enc, pred_all_sub_seq, make_prediction
from scipy.stats import mannwhitneyu
from scipy.signal import find_peaks
import numpy as np
import sys
from PARAMETERS import *
import argparse


# TODO remove
DEBUG = False

def make_all_seqs_prediction(model, seqs, max_pred=True):
    # seqs = seqs[:30]
    input_size = get_input_size(model)
    one_hot_mat_list = [one_hot_enc(s) for s in seqs]
    preds_per_seq = np.zeros(len(seqs) + 1, dtype=int)
    for i, s in enumerate(seqs):  # preds locations in the output array
        preds_per_seq[i+1] = len(s) - input_size + 1 + preds_per_seq[i]
    sub_mat_list = [np.array([m[x:x+input_size]for x in range(len(m)-input_size+1)]) for m in one_hot_mat_list]
    sub_mat_arr = np.vstack(sub_mat_list)
    # for large amount of data
    batch_size = 15000
    preds_l = []
    i = 0
    while i < len(sub_mat_arr):
        preds_l.append(make_prediction(model, one_hot_mat=sub_mat_arr[i:min(i+batch_size, len(sub_mat_arr))]))
        i += batch_size
    sub_seq_preds = np.vstack(preds_l)
    seq_preds = [sub_seq_preds[preds_per_seq[i]:preds_per_seq[i+1]] for i in range(len(preds_per_seq)-1)]
    assert len(seq_preds) == len(seqs), f"ERROR: make_all_seqs_prediction - len(preds) != len(seq)"
    return [max(p) for p in seq_preds] if max_pred else seq_preds


def predict_fasta(model, src, dst):
    with open(src) as f:
        f_lines = f.read().splitlines()
    seqs = f_lines[1::2]
    print(f"Number of sequences = {len(seqs)}")
    scores = make_all_seqs_prediction(model, seqs)
    with open(dst, "w") as f:
        f.write(",sequence,rG4detector\n")
        for s, p in zip(seqs, scores):
            f.write(f"{s},{p}\n")
    print(f"prediction avg = {sum(scores)/len(scores)}")

# TODO
def check_signification(src_1, src_2):
    stress = pd.read_csv(src_1)
    cntrl = pd.read_csv(src_2)
    U1, p = mannwhitneyu(stress["rG4detector"], cntrl["rG4detector"])
    print(f"{U1} - {p}")


def predict_screener(src, dst):
    max_preds_df = pd.DataFrame()
    screener_preds = pd.read_csv(src, usecols=["description", 'cGcC', 'G4H', 'G4NN'], sep="\t")
    seq_nums = screener_preds["description"].unique()
    for idx, num in enumerate(seq_nums):
        seq_preds = screener_preds[screener_preds["description"] == num]
        for method in ["G4H", "G4NN", "cGcC"]:
            max_preds_df.loc[idx, method] = seq_preds[method].max()
    max_preds_df.to_csv(dst, index=False)


def screener_norm(dir_path, unique):
    addToPath = "_unique" if unique else ""

    # get rg4detector norm
    rg4detector_stress_path = dir_path + f"/stress/stress_predictions{addToPath}.csv"
    rg4detector_control_path = dir_path + f"/control/control_predictions{addToPath}.csv"
    rg4detector_stress_preds = pd.read_csv(rg4detector_stress_path)["rG4detector"].to_numpy()
    rg4detector_control_preds = pd.read_csv(rg4detector_control_path)["rG4detector"].to_numpy()
    stress_len = len(rg4detector_stress_preds)
    rg4detector_preds = np.concatenate([rg4detector_stress_preds, rg4detector_control_preds])
    rg4detector_preds_norm = (rg4detector_preds - np.mean(rg4detector_preds))/np.std(rg4detector_preds)

    # get screener norm
    screener_preds_norm = {}
    screener_path = dir_path + f"/screener/{addToPath[1:]}/"
    screener_stress_path = screener_path + f"stress_screener{addToPath}_pred.csv"
    screener_control_path = screener_path + f"control_screener{addToPath}_pred.csv"
    screener_stress_preds = pd.read_csv(screener_stress_path)
    screener_control_preds = pd.read_csv(screener_control_path)
    for m in ["G4H", "G4NN", "cGcC"]:
        stress_preds = screener_stress_preds[m].to_numpy()
        control_preds = screener_control_preds[m].to_numpy()
        preds = np.concatenate([stress_preds, control_preds])
        preds_norm = (preds - np.mean(preds)) / np.std(preds)
        screener_preds_norm[m] = preds_norm

    stress_norm = pd.DataFrame()
    stress_norm["rG4detector"] = rg4detector_preds_norm[:stress_len]
    for m in ["G4H", "G4NN", "cGcC"]:
        stress_norm[m] = screener_preds_norm[m][:stress_len]
    stress_norm.to_csv(screener_path + f"stress_screener{addToPath}_norm.csv", index=False)

    control_norm = pd.DataFrame()
    control_norm["rG4detector"] = rg4detector_preds_norm[stress_len:]
    for m in ["G4H", "G4NN", "cGcC"]:
        control_norm[m] = screener_preds_norm[m][stress_len:]
    control_norm.to_csv(screener_path + f"control_screener{addToPath}_norm.csv", index=False)


def process_G3BP1_data(dir_path, model_path, unique, ensemble_size):

    if unique:
        cntrl_src = dir_path + "/control/G3BP1_2021_control_unique.fa"
        cntrl_dst = dir_path + "/control/control_predictions_unique.csv"
        stress_src = dir_path + "/stress/G3BP1_2021_stress_unique.fa"
        stress_dst = dir_path + "/stress/stress_predictions_unique.csv"
    else:
        cntrl_src = dir_path + "/control/G3BP1_2021_cntrl.fa"
        cntrl_dst = dir_path + "/control/cntrl_predictions.csv"
        stress_src = dir_path + "/stress/G3BP1_2021_stress.fa"
        stress_dst = dir_path + "/stress/stress_predictions.csv"

    MODEL = []
    for i in range(ensemble_size):
        MODEL.append(load_model(model_path + f"/model_{i}.h5"))

    # Make rG4detector predictions
    print("Starting make_preds stress")
    predict_fasta(MODEL, stress_src, stress_dst)
    print("Starting make_preds control")
    predict_fasta(MODEL, cntrl_src, cntrl_dst)


    # # get screener predictions
    # print("Predict_screener")
    # add2path = "_unique" if unique else ""
    # predict_screener(dir_path + f"/screener/{add2path[1:]}/stress_screener{add2path}.csv",
    #                  dir_path + f"/screener/{add2path[1:]}/stress_screener{add2path}_pred.csv")
    # predict_screener(dir_path + f"/screener/{add2path[1:]}/control_screener{add2path}.csv",
    #                  dir_path + f"/screener/{add2path[1:]}/control_screener{add2path}_pred.csv")

    # normalize predictions
    # print("Norm screener")
    # screener_norm(dir_path, unique)


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", dest="directory_path", help="G3BP1 data directory path", required=True)
    parser.add_argument("-m", "--model", dest="model_path", help="rG4detector model directory", required=True)
    parser.add_argument("-u", "--unique", action="store_true",
                        help="Operate in unique mode predict only on unique stress and control sequences)")
    parser.add_argument("-e", "--ensemble", dest="ensemble_size", type=int,
                        help=f"rG4detector ensemble size (default={ENSEMBLE_SIZE})", default=ENSEMBLE_SIZE)

    args = parser.parse_args()

    process_G3BP1_data(args.directory_path, args.model_path, args.unique, args.ensemble_size)






