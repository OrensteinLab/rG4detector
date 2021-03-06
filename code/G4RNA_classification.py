import sys
import time
from tensorflow.keras.models import load_model
import getopt
from utils import AUC_Score, one_hot_enc, plot_auc_curve, pred_all_sub_seq
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from PARAMETERS import *

def get_g4rna_data(g4rna_dir):
    # get x
    data_file = open(g4rna_dir + "/seq.txt")
    data = [d.upper() for d in data_file.read().splitlines()]
    x = np.array(list(map(one_hot_enc, data)), dtype=object)
    # get y
    y = pd.read_csv(g4rna_dir + "/g4rna_filtered_data.csv", usecols=["label"])
    return x, y



def main(model, g4rna_dir):
    t1 = time.time()
    prediction = {"rG4detector": [], "G4NN": [], "cGcC": [], "G4H": []}
    scores = {}

    # get data
    x, y = get_g4rna_data(g4rna_dir)
    pad_size = DATA_SIZE//2 - 15

    # pred rg4detector
    for one_hot_mat in x:
        one_hot_mat = np.vstack((np.zeros((pad_size, 4)), one_hot_mat, np.zeros((pad_size, 4))))
        prediction["rG4detector"].append(max(pred_all_sub_seq(one_hot_mat, model)))

    # pred screener
    screener_df = pd.read_csv(g4rna_dir + "/screener_preds.csv", delimiter="\t")
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

    # TODO - remove
    if PLOT:
        plot_auc_curve(scores, title="G4RNA prediction", dest=g4rna_dir + "/g4rna_roc_plot", plot=True)
    print(f"Execution time = {round((time.time()-t1))} seconds")

    # save data
    for m in scores:
        with open(g4rna_dir + f"/results/{m}_g4rna_roc.csv", "w") as f:
            f.write(f"True positive rate,False positive rate\n")
            for tpr, fpr in zip(scores[m].y, scores[m].x):
                f.write(f"{tpr},{fpr}\n")
    with open(g4rna_dir + f"/results/G4RNA_AUC.csv", "w") as f:
        f.write(f",G4RNA AUC score\n")
        for m in scores:
            f.write(f"{m},{scores[m].auc}\n")


if __name__ == "__main__":
    DEBUG = False
    PLOT = True
    if len(sys.argv) != 2:
        print("Usage: G4RNA_classification.py <g4rna_dir_path>")
        exit(0)
    g4rna_dir_path = sys.argv[1]


    models = []
    for i in range(ENSEMBLE_SIZE):
        models.append(load_model(MODEL_PATH + f"model_{i}.h5"))
    main(models, g4rna_dir_path)

















