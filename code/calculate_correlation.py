from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
from utils import get_data, get_input_size
import numpy as np
from utils import set_data_size
import pandas as pd
from PARAMETERS import *
from Bio import SeqIO


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


def get_rG4detector_mouse_corr():
    # params
    output = []
    data_rsr =  []
    data_file = pd.read_csv(MOUSE_PATH)
    records_dict = SeqIO.to_dict(SeqIO.parse("big_files/GCF_000001635_new.26_GRCm38.p6_rna.fna", "fasta"))

    for idx, row in data_file.iterrows():
        rsr_li = row["Untreated (K+)"] / row['Untreated (Li+)']
        rsr_na = row["Untreated (K+)"] / row['Untreated (Na+)']
        rsr_both = row["Untreated (K+)"] / row['Untreated (Li+/Na+)']
        transcript = row['Transcript']
        pos = row['RT-stop position']
        if transcript in records_dict and np.isfinite([rsr_li, rsr_na, rsr_both]).all():
            seq = records_dict[transcript].seq
            start = max(0, pos - 30 - (input_length-30)//2)
            end = min(len(seq), pos + (input_length-30)//2)
            head_padding = max(0, (input_length-30)//2-pos+30)
            sub_seq = str(seq[start:end])
            # print(sub_seq)
            data_rsr["Li"].append(rsr_li)
            data_rsr["Na"].append(rsr_na)
            data_rsr["Li-Na"].append(rsr_both)

            sub_mat = one_hot_enc(sub_seq)
            hot_mat = np.zeros([input_length, 4])
            hot_mat[head_padding:sub_mat.shape[0] + head_padding, :] = sub_mat
            output.append(hot_mat)
        else:
            missing_counter += 1
    print(f'\n total missing or missing data = {missing_counter} of {len(output)+missing_counter}')
    one_hot_mat = np.array(output)

    if ENSEMBLE:
        pred = np.zeros((len(one_hot_mat), 1))
        for m in model:
            pred += m(one_hot_mat).numpy()/len(model)
    else:
        pred = model.predict(one_hot_mat,  batch_size=len(one_hot_mat))
    pred = pred.reshape(len(pred))
    for t in ["Li", "Na", "Li-Na"]:
        data = data_rsr[t]
        data = np.log(data)
        corr = pearsonr(pred, data)
        # corr = spearmanr(pred, data)
        print(f'{t} correlation = {round(corr[0], 3)}')
    return round(corr[0], 3)


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


def calculate_human_correlation(model):
    print("Evaluating human correlation:")
    # get screener scores
    _,  [_, y_test], _ = get_data(DATA_PATH, min_read=2000)
    scores = get_screener_preds(file_path=SCREENER_PATH + "rg4_seq_preds.csv", y=y_test)
    scores["rG4detector"] = get_rG4detector_corr(model, DATA_PATH)
    for m in scores.keys():
        print(f"{m} Pearson correlation = {round(scores[m],3)}")


def calculate_mouse_correlation(model):


if __name__ == "__main__":
    model_path = MODEL_PATH
    rG4detector = []
    for i in range(ENSEMBLE_SIZE):
        rG4detector.append(load_model(f"{model_path}/model_{i}.h5"))
    calculate_human_correlation(rG4detector)







