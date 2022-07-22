from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
from utils import get_data, get_input_size
import numpy as np
from utils import set_data_size, one_hot_enc
import pandas as pd
from PARAMETERS import *
import argparse


def get_rG4detector_human_corr(model, data_path):
    _, [xTest, yTest, _], _ = get_data(data_path, min_read=2000)
    data_size = get_input_size(model)
    [xTest] = set_data_size(data_size, [xTest])
    preds = np.zeros((len(xTest), 1))
    for j in range(len(model)):
        preds += model[j](xTest).numpy() / len(model)
    preds = preds.reshape(len(preds))
    yTest = yTest.reshape(len(yTest))
    corr = pearsonr(preds, yTest)[0]
    return corr


def get_rG4detector_mouse_corr(model, mouse_df):
    print("Computing mouse correlation")
    input_length = get_input_size(model)
    chop_size = (len(mouse_df.loc[0, "sequence"]) - input_length) // 2
    sequences = [s[chop_size:chop_size + input_length + 1] for s in mouse_df["sequence"]]
    X = np.array(list(map(one_hot_enc, sequences)))

    pred = np.zeros((len(X), 1))
    for m in model:
        pred += m(X).numpy() / len(model)
    pred = pred.reshape(len(pred))
    log_rsr = np.log(mouse_df["label"])
    corr = pearsonr(pred, log_rsr)
    return round(corr[0], 3)


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

def calculate_mouse_correlation(model, data_path):
    print("Evaluating mouse correlation:")
    # get screener scores
    mouse_df = pd.read_csv(data_path + "mouse_data.csv", names=["sequence", "label"], header=None, delimiter="\t")
    scores = get_screener_scores(screener_preds=SCREENER_PATH + "/output_data/mouse_test_predictions.csv",
                                 y=mouse_df["label"].to_numpy())
    scores["rG4detector"] = get_rG4detector_mouse_corr(model, mouse_df)
    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_dir_path", help="Data directory path", required=True)
    parser.add_argument("-m", "--model", dest="model_path", help="rG4detector model directory", required=True)
    parser.add_argument("-e", "--ensemble", dest="ensemble_size",
                        help=f"rG4detector ensemble size (default={ENSEMBLE_SIZE})", default=ENSEMBLE_SIZE)
    args = parser.parse_args()

    rG4detector = []
    for i in range(args.ensemble_size):
        rG4detector.append(load_model(f"{args.model_path}/model_{i}.h5"))

    # calculate_human_correlation(rG4detector, args.data_dir_path + "/human/")
    calculate_mouse_correlation(rG4detector, args.data_dir_path + "/mouse/")








