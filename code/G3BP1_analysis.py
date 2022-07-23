import time
import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_input_size, one_hot_enc, pred_all_sub_seq
from scipy.stats import mannwhitneyu
from scipy.signal import find_peaks
import numpy as np
import sys
from PARAMETERS import *

DEBUG = False

def make_prediction(model, seq, max_pred=True):
    one_hot_mat = one_hot_enc(seq)
    preds = pred_all_sub_seq(one_hot_mat, model)
    if max_pred:
        return max(preds)
    else:
        return preds


def predict_fasta(model, src, dst):
    with open(src) as f:
        f_lines = f.read().splitlines()
    seqs = f_lines[1::2]
    print(f"Number of sequences = {len(seqs)}")
    scores_df = pd.DataFrame(index=range(1, len(seqs)+1), dtype=float)
    for idx, seq in enumerate(seqs):
        if (idx+1) % 500 == 0:
            print(f"{idx+1} sequences are done")
            if DEBUG:
                break
        pred = make_prediction(model, seq)
        scores_df.loc[idx+1, "sequence"] = seq
        scores_df.loc[idx+1, "rG4detector"] = pred
    scores_df.to_csv(dst)


def arrange_fasta(src):
    with open(src) as file:
        lines = file.read().splitlines()
    new_data = []
    last_is_not_header = False
    for l in lines:
        if l[0] != ">":
            if last_is_not_header:
                new_data[-1] = new_data[-1] + l
            else:
                new_data.append(l)
            last_is_not_header = True
        else:
            new_data.append(l)
            last_is_not_header = False
    file = open(src, "w")
    for line in new_data:
        file.write(line + "\n")
    file.close()


def check_signification(src_1, src_2):
    stress = pd.read_csv(src_1)
    cntrl = pd.read_csv(src_2)
    U1, p = mannwhitneyu(stress["rG4detector"], cntrl["rG4detector"])
    print(f"{U1} - {p}")


def get_best_sub_seq(model, t_hold, c_src=None, s_src=None):
    c_lines, s_lines = [], []

    input_size = get_input_size(model)
    if c_src:
        with open(c_src) as file:
            c_lines = file.read().splitlines()
    if s_src:
        with open(s_src) as file:
            s_lines = file.read().splitlines()

    seqs = c_lines[1::2] + s_lines[1::2]

    all_sub_seqs = []
    t = time.time()
    for j, seq in enumerate(seqs):
        if (j+1) % 1000 == 0:
            print(f"{j + 1}/{len(seqs)} - {round(time.time()-t)}s")
            t = time.time()
        all_sub_seqs += get_sub_seq_list(model, seq, input_size, t_hold)
    print(f"threshold = {t_hold}, number of sequences = {len(all_sub_seqs)}")
    return all_sub_seqs


def get_sub_seq_list(model, seq, input_size, t):
    one_hot_mat = one_hot_enc(seq)
    preds = pred_all_sub_seq(one_hot_mat, model)
    high_score_list = []
    for j in range(len(preds)):
        if preds[j] > t:
            high_score_list.append(j)

    flank_5_size = input_size//2 - 15
    sub_seq_list = []
    pos = 0
    while pos < len(high_score_list):
        start = high_score_list[pos] + flank_5_size
        while (pos + 1) < len(high_score_list) and (high_score_list[pos+1] - 30) < high_score_list[pos]:
            pos += 1
        end = high_score_list[pos] + flank_5_size + 30
        sub_seq = seq[start:end]
        sub_seq_list.append(sub_seq)
        pos += 1
    return sub_seq_list

# TODO
def find_seq_statistics(stress, ctrl, dst, input_size, dir_path):
    statics_df = pd.DataFrame(columns=["Condition", "Mean length", "G", "GG", "GGG"])
    for cond, src in zip(["control", "stress"], [ctrl, stress]):
        seqs_statics_df = pd.DataFrame()
        statics = {"Condition": cond, "G": 0, "GG": 0, "GGG": 0}
        # mean length
        with open(src) as f:
            f_lines = f.read().splitlines()
        seqs = f_lines[1::2]
        total_length = sum([len(x)-input_size for x in seqs])
        statics["Mean length"] = round(total_length/len(seqs), 1) - input_size

        # G content
        for idx, s in enumerate(seqs):
            trunked_seq = s[80:-50]
            seqs_statics_df.loc[idx, "sequence"] = trunked_seq
            seqs_statics_df.loc[idx, "sequence length"] = len(trunked_seq)
            for j, m in enumerate(["G", "GG", "GGG"]):
                count = trunked_seq.upper().count(m)
                seqs_statics_df.loc[idx, m] = round(count/len(trunked_seq)*(j+1), 3)
                statics[m] += trunked_seq.count(m)/total_length*(j+1)
        statics_df = statics_df.append(statics, ignore_index=True)
        statics_df.to_csv(dir_path + "statics.csv", index=False)
        seqs_statics_df.to_csv(dst + f"sequences_statics_{cond}.csv", index=False)

# TODO
def find_seq_peaks(model, src, dst, t_hold=1.4):
    print("ERROR: need to update function")
    exit(1)
    with open(src) as f:
        f_lines = f.read().splitlines()
    seqs = f_lines[1::2]

    print(f"Number of sequences = {len(seqs)}")
    peaks_df = pd.DataFrame(index=range(1, len(seqs) + 1))
    t = time.time()
    for idx, seq in enumerate(seqs):
        if (idx+1) % 1000 == 0:
            print(f"{idx + 1}/{len(seqs)} - {round(time.time()-t)}s")
            t = time.time()
        pred = make_prediction(model, seq, max_pred=False)
        p, _ = find_peaks(pred, height=t_hold)
        seq_len = len(seq) - (79 + 50)  # (pad5 + pad 3)
        peaks_df.loc[idx+1, "sequence"] = seq
        peaks_df.loc[idx+1, "rG4detector"] = max(pred)
        peaks_df.loc[idx+1, "number of predictions peaks"] = len(p)
        peaks_df.loc[idx+1, "predictions peaks locations"] = ",".join(str(x) for x in p)
        min_dst = len(seq)//2 if len(p) == 0 else min(abs(p-15-seq_len//2))
        peaks_df.loc[idx + 1, "distance from center (15 shift)"] = min_dst
    peaks_df.to_csv(dst)


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
    rg4detector_control_path = dir_path + f"/control/cntrl_predictions{addToPath}.csv"
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


def process_G3BP1_data(dir_path, model_path):
    make_preds = True
    GET_SUB_SEQ = False
    GET_STATICS = False
    FIND_PEAKS = False
    UNIQUE = True
    pred_unique = False
    SCREENER = False
    NORM = False

    if UNIQUE:
        cntrl_src = dir_path + "/control/G3BP1_2021_control_unique.fa"
        cntrl_dst = dir_path + "/control/control_predictions_unique.csv"
        stress_src = dir_path + "/stress/G3BP1_2021_stress_unique.fa"
        stress_dst = dir_path + "/stress/stress_predictions_unique.csv"
        potential_rg4_path = dir_path + "/potential_G3BP1_rG4_binding_areas_unique/"
        statics_dst = dir_path + f"/statics/unique/"
    else:
        cntrl_src = dir_path + "/control/G3BP1_2021_cntrl.fa"
        cntrl_dst = dir_path + "/control/cntrl_predictions.csv"
        stress_src = dir_path + "/stress/G3BP1_2021_stress.fa"
        stress_dst = dir_path + "/stress/stress_predictions.csv"
        potential_rg4_path = dir_path + "/potential_G3BP1_rG4_binding_areas/"
        statics_dst = dir_path + f"/statics/"

    MODEL = []
    for i in range(ENSEMBLE_SIZE):
        MODEL.append(load_model(model_path + f"/model_{i}.h5"))

    if make_preds:
        print("Starting make_preds stress")
        predict_fasta(MODEL, stress_src, stress_dst)
        print("Starting make_preds control")
        predict_fasta(MODEL, cntrl_src, cntrl_dst)

    # TODO - remove
    # if GET_SUB_SEQ:
    #     threshold = 1.64
    #     print("GET_SUB_SEQ stress")
    #     stress_sub_seqs = get_best_sub_seq(MODEL, t_hold=threshold, s_src=stress_src)
    #     print("GET_SUB_SEQ control")
    #     ctrl_sub_seqs = get_best_sub_seq(MODEL, t_hold=threshold, c_src=cntrl_src)
    #     for data_set, sub_seqs in (["control", ctrl_sub_seqs], ["stress", stress_sub_seqs]):
    #         fp = open(potential_rg4_path + f"{data_set}/potential_G3BP1_rG4_binding_area_{data_set}_{threshold}.txt",
    #                   "w")
    #         for counter, line in enumerate(sub_seqs):
    #             fp.write(f">{counter}\n" + line + "\n")
    #         fp.close()

    # if GET_STATICS:
    #     print("get statics start")
    #     find_seq_statistics(stress=stress_src, ctrl=cntrl_src, dst=statics_dst)

    # TODO - remove
    # if FIND_PEAKS:
    #     threshold = 1.64
    #     print("FIND_PEAKS start")
    #     find_seq_peaks(MODEL, stress_src, peak_dest + "stress_peaks.csv", t_hold=threshold)
    #     find_seq_peaks(MODEL, cntrl_src, peak_dest + "control_peaks.csv", t_hold=threshold)

    if pred_unique:
        print("Starting stress unique")
        predict_fasta(MODEL, dir_path + "G3BP1_2021_stress_unique.fa", dir_path + "stress_unique_preds.csv")
        print("Starting common")
        predict_fasta(MODEL, dir_path + "G3BP1_2021_common.fa", dir_path + "common_preds.csv")
        print("Starting control unique")
        predict_fasta(MODEL, dir_path + "G3BP1_2021_control_unique.fa", dir_path + "control_unique_preds.csv")

    if SCREENER:
        print("Predict_screener")
        add2path = "_unique" if UNIQUE else ""
        predict_screener(dir_path + f"screener/{add2path[1:]}/stress_screener{add2path}.csv",
                         dir_path + f"screener/{add2path[1:]}/stress_screener{add2path}_pred.csv")
        predict_screener(dir_path + f"screener/{add2path[1:]}/control_screener{add2path}.csv",
                         dir_path + f"screener/{add2path[1:]}/control_screener{add2path}_pred.csv")

    if NORM:
        print("Norm screener")
        screener_norm(dir_path, UNIQUE)


if __name__ == "__main__":

    directory_path = sys.argv[1]
    model_dir_path = sys.argv[2]
    process_G3BP1_data(directory_path, model_dir_path)






