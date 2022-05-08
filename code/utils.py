import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from one_hot import one_hot_enc
from scipy.stats import pearsonr
import numpy as np
from tensorflow.keras.utils import to_categorical
from PARAMETERS import *


def set_data_size(data_size, data_sets):
    total_data_size = data_sets[0].shape[1]
    start = total_data_size//2 - data_size//2
    end = start + data_size
    return_list = []
    for data_set in data_sets:
        return_list.append(data_set[:, start:end, :])
    return return_list

# class AUC_Score:
#     def __init__(self, method, x, y, auc):
#         self.method = method
#         self.x = x
#         self.y = y
#         self.auc = auc
#


def get_data(path, min_read=2000):
    # train
    with open(DATA_PATH + f'/seq/train-seq') as source:
        X_train = np.array(list(map(one_hot_enc, source)))
    y_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['rsr']).to_numpy()

    # validation
    with open(DATA_PATH + f'/seq/val-seq') as source:
        X_val = np.array(list(map(one_hot_enc, source)))
    y_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['rsr']).to_numpy()
    w_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['c_read']).to_numpy() +\
            pd.read_csv(path + '/csv_data/val_data.csv', usecols=['t_read']).to_numpy()
    # test
    with open(DATA_PATH + f'/seq/test-seq') as source:
        X_test = np.array(list(map(one_hot_enc, source)))
    y_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['rsr']).to_numpy()
    w_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['c_read']).to_numpy() + \
             pd.read_csv(path + '/csv_data/test_data.csv', usecols=['t_read']).to_numpy()

    # set test and val min read
    ids = np.argwhere(w_test > min_read)[:, 0]
    X_test = X_test[ids]
    y_test = y_test[ids]
    ids = np.argwhere(w_val > min_read)[:, 0]
    X_val = X_val[ids]
    y_val = y_val[ids]

    # scale_labels
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    y_val = np.log(y_val)
    return [X_train, y_train], [X_test, y_test], [X_val, y_val]
#
#
# def print_results(corr_list, start_time):
#     corr = sum(corr_list) / len(corr_list)
#     corr_std = np.std(corr_list)
#     print(f'corr = {corr} +- {corr_std}')
#     # logging.info(f'w_corr = {w_corr}')
#     print("Finished Level - execution time = %ss ---\n\n" % (round(time.time() - start_time)))
#
#
# def plot_data(Y, level):
#     plt.boxplot(Y)
#     plt.title(f'fsr score distribution - min read = {level}')
#     plt.savefig(f'/home/maor/rG4/rg4detector/fsr_data/k/data_distribution/{level}.png')
#     plt.show()


def get_input_size(model):
    bagging = True if isinstance(model, list) else False

    if bagging:
        if len(model[0].layers[0].input_shape) == 1:
            input_size = model[0].layers[0].input_shape[0][1]
        else:
            input_size = model[0].layers[0].input_shape[1]
    else:
        if len(model.layers[0].input_shape) == 1:
            input_size = model.layers[0].input_shape[0][1]
        else:
            input_size = model.layers[0].input_shape[1]
    return input_size
#
#
# def make_prediction(model, seq=None, one_hot_mat=None):
#     if one_hot_mat is None:
#         if isinstance(seq, list):
#             one_hot_mat = np.array(list(map(one_hot_enc, seq)))
#         else:
#             one_hot_mat = one_hot_enc(seq)
#             one_hot_mat = one_hot_mat.reshape((1, one_hot_mat.shape[0], one_hot_mat.shape[1]))
#     if isinstance(model, list):
#         pred = np.zeros((len(one_hot_mat), 1))
#         for m in range(len(model)):
#             pred += model[m](one_hot_mat).numpy() / len(model)
#     else:
#         pred = model.predict(one_hot_mat)
#     return pred
#
#
# def plot_auc_curve(scores_dict, title=None, dest=None, plot=False, PR=False, y=None):
#     legend_list = []
#     for method in scores_dict:
#         plt.plot(scores_dict[method].x, scores_dict[method].y)  # label=scores_dict[method].method)
#         legend_list.append(f"{method} - AUC = {scores_dict[method].auc}")
#     if PR:
#         plt.xlabel("Recall")
#         plt.ylabel("Precision")
#         baseline = sum(y) / len(y)
#         plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
#     else:
#         plt.xlabel("False-Positive Rate")
#         plt.ylabel("True-Positive Rate")
#         plt.plot([0, 1], [0, 1], linestyle="--")
#         legend_list.append("Random Guess")
#     plt.legend(legend_list)
#     if title:
#         plt.title(title)
#     if dest:
#         plt.savefig(dest)
#     if plot:
#         plt.show()
#
#
# def get_screener_scores(file_path, y):
#     scores = {}
#     pred = pd.read_csv(file_path, usecols=['cGcC', 'G4H', 'G4NN'], sep="\t")
#     labels = y.reshape(len(y))
#     for col in pred.columns:
#         const = 0
#         preds = pred[col].to_numpy()
#         if min(preds) < 0:
#             const = -min(preds) + 10 ** -3
#         preds = preds + const
#         preds = np.log(preds)
#         pr, p = pearsonr(preds, labels)
#         print(f'\n{col}:')
#         print(f"corr = {round(pr, 3)}, p_value = {round(p, 3)}")
#         scores[col] = round(pr, 3)
#     return scores


def one_hot_enc(s, remove_last=True):
    if remove_last:
        s = s[:-1]
    s = s + "ACGT"
    if 'N' not in s and 'Z' not in s and 'H' not in s:
        trans = s.maketrans('ACGT', '0123')
        numSeq = list(s.translate(trans))
        return to_categorical(numSeq)[0:-4]
    else:
        s = s + "NZH"
        trans = s.maketrans('ACGTNZH', '0123456')
        numSeq = list(s.translate(trans))
        hotVec = to_categorical(numSeq)[0:-7]
        for i in range(len(hotVec)):
            if hotVec[i][4] == 1:
                hotVec[i] = [0.25, 0.25, 0.25, 0.25, 0, 0, 0]
            if hotVec[i][5] == 1:
                hotVec[i] = [0, 0, 0, 0, 0, 0, 0]
            if hotVec[i][6] == 1:
                hotVec[i] = [1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0]
        return np.delete(hotVec, [4, 5, 6], 1)


def pred_all_sub_seq(data, model):
    input_length = get_input_size(model)
    sub_seq_list = np.array([data[x:x+input_length, :] for x in range(len(data)-input_length+1)])

    if isinstance(model, list):
        preds = np.zeros((len(sub_seq_list), 1))
        for i in range(len(model)):
            preds += model[i](sub_seq_list).numpy()/len(model)
    else:
        preds = model(sub_seq_list).numpy()
    preds = preds.reshape(len(preds))
    return preds

def get_score_per_position(preds, input_length, sigma):
    positions_scores = np.zeros((len(preds)-input_length+1))  # number of sigmas X transcript size
    gaussian_filter = get_gaussian_filter(input_length, sigma=sigma)
    positions_scores[:] = np.convolve(preds, gaussian_filter, "valid")
    return positions_scores

def get_gaussian_filter(input_size, sigma=20):
    gaussian_filter = np.zeros(input_size)
    counter = 0
    for x in range(-input_size // 2, input_size // 2 + 1):
        if x == 0:
            continue
        gaussian_filter[counter] = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (2 * np.pi * (sigma ** 2))
        counter += 1
    gaussian_filter = gaussian_filter / sum(gaussian_filter)
    return gaussian_filter


class PRScore:
    def __init__(self, method, precision, recall, threshold, aucpr):
        self.method = method
        self.precision = precision
        self.recall = recall
        self.threshold = threshold
        self.auc = aucpr

