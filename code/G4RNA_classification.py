import sys
import time
from tensorflow.keras.models import load_model
import os.path
import getopt
from datetime import datetime
from utils import AUC_Score, one_hot_enc, plot_auc_curve, get_input_size, pred_all_sub_seq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from sklearn.metrics import roc_curve, auc

def get_g4rna_data():
    # get x
    data_file = open("G4RNA/seq.txt")
    data = [d.upper() for d in data_file.read().splitlines()]
    x = np.array(list(map(one_hot_enc, data)), dtype=object)
    # get y
    y = pd.read_csv("G4RNA/g4rna_filtered_data.csv", usecols=["label"])
    return x, y


def plot_scores(scores_dict, y, plot=False):
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M")
    dest = output + f"/plots/"
    if path.exists(dest) is False:
        os.makedirs(dest)
    # legend_list = []
    for method in scores_dict:
        plt.plot(scores_dict[method].recall[1:], scores_dict[method].precision[1:],
                 label=f"{method} - {round(scores_dict[method].auc, 3)}")
    # plot baseline
    baseline = sum(y)/len(y)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')

    plt.legend()
    plt.title(f"G4RNA AUC-PR")
    plt.xlabel("Recall")
    # plt.ylim([0.6, 1.05])
    plt.ylabel("Precision")
    plt.savefig(dest + f"G4RNA_AUCPR_{dt_string}")
    if PLOT:
        plt.show()


def main(model):
    t1 = time.time()
    prediction = {"rG4detector": [], "G4NN": [], "cGcC": [], "G4H": []}
    scores = {}

    # get data
    x, y = get_g4rna_data()
    pad_size = get_input_size(model)//2 - 15

    # pred rg4detector
    for one_hot_mat in x:
        one_hot_mat = np.vstack((np.zeros((pad_size, 4)), one_hot_mat, np.zeros((pad_size, 4))))
        prediction["rG4detector"].append(max(pred_all_sub_seq(one_hot_mat, model)))

    # pred screener
    screener_df = pd.read_csv(screener_path, delimiter="\t")
    seqs_ids = screener_df["description"].unique()
    for seq_id in seqs_ids:
        seq_preds = screener_df[screener_df["description"] == seq_id]
        for method in ["G4NN", "cGcC", "G4H"]:
            prediction[method].append(seq_preds[method].max())

    for method in prediction.keys():
        fpr, tpr, _ = roc_curve(y, prediction[method])
        roc_auc = auc(fpr, tpr)
        print(f"{method} roc_auc = {roc_auc}")
        scores[method] = AUC_Score(method=method, y=tpr, x=fpr, auc=round(roc_auc, 3))

    plot_auc_curve(scores, title="G4RNA prediction", dest=model_path + "g4rna_roc_plot", plot=True)
    print(f"exe time = {round((time.time()-t1))} seconds")

    # save data
    for method in scores.keys():
        roc_df = pd.DataFrame()
        roc_df["True positive rate"] = scores[method].y
        roc_df["False positive rate"] = scores[method].x
        roc_df.to_csv(model_path + f"/G4RNA/{method}_g4rna_roc.csv")


if __name__ == "__main__":
    DEBUG = False
    PLOT = True
    ensemble_size = 5
    KPDS = False
    model_path = "models/best_model/ensemble/"
    screener_path = "G4RNA/screener_preds.csv"
    if KPDS:
        model_path = "kpds/" + model_path

    opts, args = getopt.getopt(sys.argv[1:], 'dpgcsi:m:')
    for op, val in opts:
        if op == "-d":
            DEBUG = True
        if op == "-p":
            PLOT = True
        if op == "-i":
            dir_path = val

    output = model_path + "/G4RNA/"
    if path.exists(output) is False:
        os.makedirs(output)

    print(f"DEBUG is {DEBUG}")
    print(f"PLOT is {PLOT}")
    print(f"output = {output}")
    print(f"model_path = {model_path}")

    models = []
    for i in range(ensemble_size):
        models.append(load_model(f"{model_path}/model_{i}.h5"))

    main(models)

















