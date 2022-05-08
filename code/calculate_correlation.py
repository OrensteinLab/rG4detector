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




def get_screener_scores(file_path, y):
    screener_scores = {}
    pred = pd.read_csv(file_path, usecols=METHODS_LIST, sep="\t")

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


def calculate_human_correlation(model):
    print("Evaluating human correlation:")
    # get screener scores
    _,  [_, y_test], _ = get_data(DATA_PATH, min_read=2000)
    scores = get_screener_scores(file_path=SCREENER_PATH + "screener_human_preds.csv", y=y_test)
    scores["rG4detector"] = get_rG4detector_human_corr(model, DATA_PATH)
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
    calculate_human_correlation(rG4detector)








