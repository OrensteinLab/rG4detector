from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
from utils import get_data, get_input_size
import numpy as np
from utils import set_data_size, one_hot_enc
import pandas as pd
from PARAMETERS import *
from Bio import SeqIO
import getopt
import sys


def get_rG4detector_human_corr(model, path):
    _, [xTest, yTest], [_, _] = get_data(path, min_read=2000)
    data_size = get_input_size(model)
    [xTest] = set_data_size(data_size, [xTest])
    preds = np.zeros((len(xTest), 1))
    for j in range(len(model)):
        preds += model[j](xTest).numpy() / len(model)
    preds = preds.reshape(len(preds))
    yTest = yTest.reshape(len(yTest))
    corr = pearsonr(preds, yTest)[0]
    return corr


def get_rG4detector_mouse_corr(model, transcriptome_path):
    data_size = get_input_size(model)
    # params
    output = []
    rts_data = []
    seqs = []
    data_file = pd.read_csv(MOUSE_PATH)
    records_dict = SeqIO.to_dict(SeqIO.parse(transcriptome_path, "fasta"))

    for idx, row in data_file.iterrows():
        rts_ratio = row["Untreated (K+)"] / row['Untreated (Li+)']
        transcript = row['Transcript']
        pos = row['RT-stop position']
        if transcript in records_dict and np.isfinite(rts_ratio):
            seqs.append(row["Sequence upstream of RT stop"])
            seq = records_dict[transcript].seq
            start = max(0, pos - 30 - (data_size-30)//2)
            end = min(len(seq), pos + (data_size-30)//2)
            head_padding = max(0, (data_size-30)//2-pos+30)
            sub_seq = str(seq[start:end])
            rts_data.append(rts_ratio)
            sub_mat = one_hot_enc(sub_seq, remove_last=False)
            hot_mat = np.zeros([data_size, 4])
            hot_mat[head_padding:sub_mat.shape[0] + head_padding, :] = sub_mat
            output.append(hot_mat)
    one_hot_mat = np.array(output)

    rts_ratio = np.log(rts_data)
    scores = get_screener_scores(file_path=SCREENER_PATH + "screener_mouse_preds.csv", y=rts_ratio)
    pred = np.zeros((len(one_hot_mat), 1))
    for m in model:
        pred += m(one_hot_mat).numpy()/len(model)
    pred = pred.reshape(len(pred))
    scores["rG4detector"] = pearsonr(pred, rts_ratio)
    return scores


def get_screener_scores(file_path, y):
    screener_scores = {}
    pred = pd.read_csv(file_path, usecols=['description', 'cGcC', 'G4H', 'G4NN'], sep="\t")
    pred = pred.groupby(['description'], sort=False).max()

    labels = y.reshape(len(y))
    for col in pred.columns:
        const = 0
        preds = pred[col].to_numpy()
        if min(preds) < 0:
            const = -min(preds) + 10 ** -3
        preds = preds + const
        preds = np.log(preds)
        pr, p = pearsonr(preds, labels)
        screener_scores[col] = round(pr, 3)
    return screener_scores


def calculate_human_correlation(model):
    print("Evaluating human correlation:")
    # get screener scores
    _,  [_, y_test], _ = get_data(DATA_PATH, min_read=2000)
    scores = get_screener_scores(file_path=SCREENER_PATH + "screener_human_preds.csv", y=y_test)
    scores["rG4detector"] = get_rG4detector_human_corr(model, DATA_PATH)
    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


def calculate_mouse_correlation(model):
    print("Evaluating mouse correlation:")
    # get screener scores
    # rts_data = []
    # data_file = pd.read_csv(MOUSE_PATH)
    # for idx, row in data_file.iterrows():
    #     rts_ratio = row["Untreated (K+)"] / row['Untreated (Li+)']
    #     rts_data.append(rts_ratio)
    # scores = get_screener_scores(file_path=SCREENER_PATH + "screener_mouse_preds.csv", y=np.log(rts_data))
    scores = get_rG4detector_mouse_corr(model, MOUSE_TRANSCRIPTOME_PATH)
    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


if __name__ == "__main__":
    model_path = MODEL_PATH
    rG4detector = []
    for i in range(ENSEMBLE_SIZE):
        rG4detector.append(load_model(f"{model_path}/model_{i}.h5"))

    # opts, args = getopt.getopt(sys.argv[1:], 'hma')
    # for op, val in opts:
    #     if op == "-h":
    # calculate_human_correlation(rG4detector)
        # if op == "-m":
        #     calculate_mouse_correlation(rG4detector)
    calculate_mouse_correlation(rG4detector)







