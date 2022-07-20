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
import pickle


def get_rG4detector_human_corr(model, path):
    _, [xTest, yTest, _], _ = get_data(path, min_read=2000)
    data_size = get_input_size(model)
    [xTest] = set_data_size(data_size, [xTest])
    preds = np.zeros((len(xTest), 1))
    for j in range(len(model)):
        preds += model[j](xTest).numpy() / len(model)
    preds = preds.reshape(len(preds))
    yTest = yTest.reshape(len(yTest))
    corr = pearsonr(preds, yTest)[0]
    return corr


def get_screener_scores(screener_preds, y):
    screener_scores = {}
    pred = pd.read_csv(screener_preds, usecols=METHODS_LIST, sep="\t")

    labels = y.reshape(len(y))
    for col in pred.columns:
        const = 0
        preds = pred[col].to_numpy()

        if min(preds) <= 0:
            const = -min(preds) + 10 ** -3
        preds = preds + const
        preds = np.log(preds)
        pr, p = pearsonr(preds, labels)
        screener_scores[col] = round(pr, 3)
    return screener_scores


def calculate_human_correlation(model, data_path):
    print("Evaluating human correlation:")
    # get screener scores
    _,  [_, y_test, _], _ = get_data(data_path, min_read=2000)
    scores = get_screener_scores(screener_preds=SCREENER_PATH + "/output_data/human_test_predictions.csv", y=y_test)
    scores["rG4detector"] = get_rG4detector_human_corr(model, data_path)
    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: calculate_correlation <model_dir_path> <data_dir_path>")
        exit(0)
    model_path = sys.argv[1]
    data_dir_path = sys.argv[2]
    ens_size = ENSEMBLE_SIZE
    if len(sys.argv) == 4:
        ens_size = sys.argv[3]


    rG4detector = []
    for i in range(ens_size):
        rG4detector.append(load_model(f"{model_path}/model_{i}.h5"))

    calculate_human_correlation(rG4detector, data_dir_path)








