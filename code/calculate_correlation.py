from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
from utils import get_data, get_input_size
import numpy as np
from utils import set_data_size
import pandas as pd
from PARAMETERS import *


def get_rG4detector_corr(model, path):
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


def get_screener_preds(file_path, y):
    screener_scores = {}
    pred = pd.read_csv(file_path, usecols=['cGcC', 'G4H', 'G4NN'], sep="\t")
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


if __name__ == "__main__":
    model_path = MODEL_PATH
    data_path = DATA_PATH
    rG4detector = []

    # get screener scores
    _,  [_, y_test], _ = get_data(data_path, min_read=2000)
    scores = get_screener_preds(file_path=SCREENER_PATH + "rg4_seq_preds.csv", y=y_test)

    for i in range(ENSEMBLE_SIZE):
        rG4detector.append(load_model(f"{model_path}/model_{i}.h5"))
    scores["rG4detector"] = get_rG4detector_corr(rG4detector, data_path)

    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


