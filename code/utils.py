import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical


def set_data_size(data_size, data_sets):
    total_data_size = data_sets[0].shape[1]
    start = total_data_size//2 - data_size//2
    end = start + data_size
    return_list = []
    for data_set in data_sets:
        return_list.append(data_set[:, start:end, :])
    return return_list


def get_data_from_file(get_seq, file_path, dataset):
    data_file = open(file_path + f"/seq/{dataset}-seq", "r")
    seqs = data_file.read().splitlines()
    X = np.array(list(map(one_hot_enc, seqs))) if not get_seq else np.array(seqs)
    y = pd.read_csv(file_path + f'/csv_data/{dataset}_tr_data.csv', usecols=['rsr']).to_numpy()
    w = pd.read_csv(file_path + f'/csv_data/{dataset}_tr_data.csv', usecols=['total_reads']).to_numpy()
    return X, y, w


def get_data(path, min_read=2000, get_seq=False):
    # train
    X_train, y_train, w_train = get_data_from_file(get_seq, file_path=path, dataset="train")
    # validation
    X_val, y_val, w_val = get_data_from_file(get_seq, file_path=path, dataset="val")
    # test
    X_test, y_test, w_test = get_data_from_file(get_seq, file_path=path, dataset="test")

    # set test and val min read
    ids = np.argwhere(w_val > min_read)[:, 0]
    X_val = X_val[ids]
    y_val = y_val[ids]
    w_val = w_val[ids]
    ids = np.argwhere(w_test > min_read)[:, 0]
    X_test = X_test[ids]
    y_test = y_test[ids]
    w_test = w_test[ids]

    # scale_labels
    y_train = np.log(y_train)
    y_val = np.log(y_val)
    y_test = np.log(y_test)
    return [X_train, y_train, w_train], [X_test, y_test, w_test], [X_val, y_val, w_val]



def get_input_size(model):
    ensemble = True if isinstance(model, list) else False

    if ensemble:
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


def make_prediction(model, seq=None, one_hot_mat=None):
    if one_hot_mat is None:
        if isinstance(seq, list):
            one_hot_mat = np.array(list(map(one_hot_enc, seq)))
        else:
            one_hot_mat = one_hot_enc(seq)
            one_hot_mat = one_hot_mat.reshape((1, one_hot_mat.shape[0], one_hot_mat.shape[1]))
    if isinstance(model, list):
        pred = np.zeros((len(one_hot_mat), 1))
        for m in range(len(model)):
            pred += model[m](one_hot_mat).numpy() / len(model)
    else:
        pred = model.predict(one_hot_mat)
    return pred


def plot_auc_curve(scores_dict, title=None, dest=None, plot=False, PR=False, y=None):
    legend_list = []
    for method in scores_dict:
        plt.plot(scores_dict[method].x, scores_dict[method].y)
        legend_list.append(f"{method} - AUC = {scores_dict[method].auc}")
    if PR:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        baseline = sum(y) / len(y)
        plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    else:
        plt.xlabel("False-Positive Rate")
        plt.ylabel("True-Positive Rate")
        plt.plot([0, 1], [0, 1], linestyle="--")
        legend_list.append("Random Guess")
    plt.legend(legend_list)
    if title:
        plt.title(title)
    if dest:
        plt.savefig(dest)
    if plot:
        plt.show()


def one_hot_enc(s):
    if s[-1] == "\n":
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


def pred_all_sub_seq(data, model, pad=False):
    input_length = get_input_size(model)
    if pad:
        data = np.concatenate([np.zeros((input_length-1, 4)), data, np.zeros((input_length-1, 4))])
    sub_seq_arr = np.array([data[x:x+input_length, :] for x in range(len(data)-input_length+1)])

    if isinstance(model, list):
        preds = np.zeros((len(sub_seq_arr), 1))
        for i in range(len(model)):
            preds += model[i](sub_seq_arr).numpy()/len(model)
    else:
        preds = model(sub_seq_arr).numpy()
    preds = preds.reshape(len(preds))
    return preds


def get_score_per_position(preds, input_length, sigma):
    positions_scores = np.zeros((len(preds)-input_length+1))
    gaussian_filter = get_gaussian_filter(input_length, sigma=sigma)
    positions_scores[:] = np.convolve(preds, gaussian_filter, "valid")
    return positions_scores


def get_gaussian_filter(input_size, sigma=20, mu=15):
    gaussian_filter = np.zeros(input_size)
    counter = 0
    for x in range(-input_size // 2, input_size // 2 + 1):
        if x == 0:
            continue
        gaussian_filter[counter] = np.exp(-((x-mu) ** 2) / (2 * sigma ** 2))
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

class AUC_Score:
    def __init__(self, method, x, y, auc):
        self.method = method
        self.x = x
        self.y = y
        self.auc = auc


def set_screener_positions_scores(screener_scores):
    SCREENER_WINDOW_LENGTH = 29
    screener_positions_score = {}
    for method in screener_scores:
        positions_scores = np.zeros((len(screener_scores[method]) + SCREENER_WINDOW_LENGTH - 1, 1))
        for i in range(len(screener_scores[method]) + SCREENER_WINDOW_LENGTH - 1):
            start = max(0, i - SCREENER_WINDOW_LENGTH + 1)
            end = min(len(screener_scores[method]), i + SCREENER_WINDOW_LENGTH)
            positions_scores[i] = max(screener_scores[method][start:end])
        # positions_scores = positions_scores.reshape(len(positions_scores), 1)
        screener_positions_score[method] = positions_scores
    return screener_positions_score

