import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def set_data_size(data_size, data_sets):
    """
    Sets data sequences to the desired lenth
    :param data_size: Sequences length (int)
    :param data_sets: List containing array-like of shape (n_samples, seq_length, 4) datasets
    :return: list oth the choped datasets
    """
    total_data_size = data_sets[0].shape[1]
    start = total_data_size//2 - data_size//2
    end = start + data_size
    return_list = []
    for data_set in data_sets:
        return_list.append(data_set[:, start:end, :])
    return return_list


def get_data_from_file(get_seq, file_path, dataset):
    """
    Read the sequences, labels and weights for a single dataset

    :param get_seq: boolean - get sequences (True) or one-hot-encoded matrix
    :param file_path: str- file_path:path to source directory
    :param dataset: str - dataset (train/val/test)
    :return: X, y, w (Data, labels, weights)
    """
    data_file = open(file_path + f"/seq/{dataset}-seq", "r")
    seqs = data_file.read().splitlines()
    X = np.array(list(map(one_hot_enc, seqs))) if not get_seq else np.array(seqs)
    y = pd.read_csv(file_path + f'/csv_data/{dataset}_tr_data.csv', usecols=['rsr']).to_numpy()
    w = pd.read_csv(file_path + f'/csv_data/{dataset}_tr_data.csv', usecols=['total_reads']).to_numpy()
    return X, y, w


def get_data(path, min_read=2000, get_seq=False):
    """
    Read the sequences, labels and weights from the source directory

    :param path: str - file_path:path to source directory
    :param min_read: int - filter test and val with less than min_read read count
    :param get_seq: boolean - get sequences (True) or one-hot-encoded matrix
    :return: (train, val, tests) lists of (data, labels, weights)
    """
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
    """
    Gets the model input size

    :param model: model object or a list of models
    :return: input_size
    """
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
    """
    Makes prediction of a given sequences/matrix

    :param model: model object or a list of models
    :param seq: list of sequences or None
    :param one_hot_mat: array-like of shape (n_samples, seq_length, 4) or Nona
    :return:
    """
    if one_hot_mat is None:
        assert seq is not None, f"Both seq and one_hot_mat are None, need to pass at least one object"
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
    """
    Plots ROC/PR curves for a given predictions

    :param scores_dict: Dictionary of AUC_Score objects
    :param title: str -Plot title (Optional)
    :param dest: str -Plot save destination(Optional)
    :param plot: boolean - show (True) plot (Optional)
    :param PR: boolean - AUPR (True) or AUROC (False-default)
    :param y: For random guess calculation in AUPR (Optional)
    """
    legend_list = []
    scores_dict["G4Hunter"] = scores_dict["G4H"]
    scores_dict["cGcC-scoring"] = scores_dict["cGcC"]
    del scores_dict["cGcC"]
    del scores_dict["G4H"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in scores_dict:
        plt.plot(scores_dict[method].x, scores_dict[method].y)
        legend_list.append(f"{method} - AUC = {scores_dict[method].auc}")
    if PR:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        baseline = sum(y) / len(y)
        plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    else:
        plt.grid(alpha=0.3)
        plt.xlabel("False-positive rate", fontsize=12)
        plt.ylabel("True-positive rate", fontsize=12)
        plt.plot([0, 1], [0, 1], linestyle="--")
        legend_list.append("Random Guess")
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(legend_list)
    if title:
        plt.title(title)
    if dest:
        plt.savefig(dest)
    if plot:
        plt.show()



def one_hot_enc(s):
    """
    One hot encoding function

    :param s: str - DNA - sequences
    :return: np.array - one hoe encoded matrix
    """
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
    """
    Predicts all sub-sequences with window size of the model input and with step size of 1 in a geiven one-hot-encoded
    matrix

    :param data: np.array - one-hot-encoded matrix
    :param model: model object or a list of models
    :param pad: boolean - zero padding
    :return: all sub-sequences predictions
    """
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
    """
    Computes the detection score per position using gaussian weighted average

    :param preds: np.array - all sub sequences predictions
    :param input_length: int - model input size
    :param sigma: flout - gaussian sigma
    :return: positions_scores - np.array - final prediction per poistion
    """
    positions_scores = np.zeros((len(preds)-input_length+1))
    gaussian_filter = get_gaussian_filter(input_length, sigma=sigma, mu=11)
    positions_scores[:] = np.convolve(preds, gaussian_filter, "valid")
    return positions_scores


def get_gaussian_filter(input_size, sigma=20, mu=0):
    """
    Returns gaussian filter

    :param input_size: int - model input size
    :param sigma: flout - gaussian filter sigma
    :param mu: int - gaussian filter mean
    :return: gaussian_filter
    """
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
    """
    Class for saving PR values for detection
    """
    def __init__(self, method, precision, recall, threshold, aucpr):
        self.method = method
        self.precision = precision
        self.recall = recall
        self.threshold = threshold
        self.auc = aucpr

class AUC_Score:
    """
    Class for saving ROC values for detection
    """
    def __init__(self, method, x, y, auc):
        self.method = method
        self.x = x
        self.y = y
        self.auc = auc


def set_screener_positions_scores(screener_scores, gaussian=True, average=False, window_size=80):
    """
    Calculates RNA screener methods detection positions' scores

    :param screener_scores: Dictionary of 1D-arrays of of the RNA screener methods predictions for a given transcript
    :param gaussian:
    :param average:
    :param window_size:
    :return:
    """
    screener_positions_score = {}
    for method in screener_scores:
        if method == "G4H":
            positions_scores = screener_scores[method]
        elif gaussian:
            pad = np.zeros(window_size-1)
            pred = np.hstack((pad, screener_scores[method], pad))
            gaussian_filter = get_gaussian_filter(window_size, sigma=8)
            positions_scores = np.convolve(pred, gaussian_filter, "valid")
        else:
            positions_scores = np.zeros((len(screener_scores[method]) + window_size - 1))
            for i in range(len(screener_scores[method]) + window_size - 1):
                start = max(0, i - window_size + 1)
                end = min(len(screener_scores[method]), i + window_size)
                if average:
                    positions_scores[i] = sum(screener_scores[method][start:end])/len(screener_scores[method][start:end])
                else:
                    positions_scores[i] = max(screener_scores[method][start:end])
        screener_positions_score[method] = positions_scores
    return screener_positions_score


class transcript:
    """
    Transcript details, from annotation file parsing
    This class support the process of the determination of the most prominent transcript
    """
    def __init__(self, name, length, start, end, tsl, gene, strand):
        self.name = name
        self.len = length
        self.exons_ranges = []
        self.start = int(start)
        self.end = int(end)
        self.gene = gene
        self.tsl = 6 if tsl == "NA" else int(tsl)
        self.strand = strand


    def add_exon(self, start, end):
        pos = 0
        start, end = int(start), int(end)
        while pos < len(self.exons_ranges):
            if self.strand == "+":
                if start < self.exons_ranges[pos][0]:
                    break
            else:
                if start > self.exons_ranges[pos][0]:
                    break
            pos += 1
        self.exons_ranges.insert(pos, (start, end))

    def __str__(self):
        print_str = f"name = {self.name}\nlength = {self.len}\nstart = {self.start}\nend = {self.end}\n" \
                    f"tsl = {self.tsl}\nexons = {self.exons_ranges}"
        return print_str


def label_sequence(seq, preds, t, screener_window):
    """
    Labels each position in a given transcript according to G4Hunter prediction
    :param seq: str - RNA sequence
    :param preds: np.array - G4Hunter predictions
    :param t: flout - threshold
    :param screener_window: int - window size
    :return: final_pred - sequence labels
    """
    s = 0
    final_pred = np.zeros(len(seq))

    # find interval above threshold and label the as one
    while s < len(preds):
        while s < len(preds) and preds[s] < t:
            s += 1
        if s == len(preds):
            break

        e = s
        while e < len(preds) and preds[e] > t:
            e += 1
        e -= 1
        start = s
        end = e + screener_window - 1


        # label 0 sequences with no G
        if "G" not in seq[start:end]:
            final_pred[start:end + 1] = 0
            s = e + 1
            continue

        # remove non-Gs nt at the sequence ends
        while seq[start] != "G":
            start += 1

        while seq[end] != "G":
            end -= 1

        final_pred[start:end+1] = 1
        s = e + 1
    return final_pred


def get_G4Hunter_roc(sequences, predictions_l, thresholds, ground_truth, screener_window):
    """
    G4Hunter algorithm for detecting rG4 transcriptom wide

    :param sequences:
    :param predictions_l:
    :param thresholds:
    :param ground_truth:
    :param screener_window:
    :return:
    """
    precision, recall = [], []
    for j, t in enumerate(thresholds):
        # get labeled sequences by threshold
        TP, FP, FN = 0, 0, 0
        for i, trans in enumerate(ground_truth):
            labeled_sequence = label_sequence(str(sequences[trans].seq), predictions_l[i], t, screener_window)
            # calculate TP and FP
            TP += np.sum(np.logical_and(ground_truth[trans], labeled_sequence == 1))
            FP += np.sum(np.logical_and(np.logical_not(ground_truth[trans]), labeled_sequence == 1))
            FN += np.sum(np.logical_and(ground_truth[trans], labeled_sequence == 0))
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
    return precision, recall


def make_all_seqs_prediction(model, seqs, max_pred=True, pad=False, verbose=0):
    """
    Makes all predictions of a given list of RNA sequences

    :param model: model object or list of models
    :param seqs: list of sequences
    :param max_pred: boolean - return the maximum prediction per sequence
    :param pad: pad value - Z/N - if False - no padding added
    :return: sequences' predictions
    """
    # seqs = seqs[:30]
    input_size = get_input_size(model)
    if pad:
        seqs = [pad * (input_size - 1) + s + pad * (input_size - 1) for s in seqs]
    one_hot_mat_list = [one_hot_enc(s) for s in seqs]
    preds_per_seq = np.zeros(len(seqs) + 1, dtype=int)
    for i, s in enumerate(seqs):  # preds locations in the output array
        preds_per_seq[i+1] = len(s) - input_size + 1 + preds_per_seq[i]
    sub_mat_list = [np.array([m[x:x+input_size]for x in range(len(m)-input_size+1)]) for m in one_hot_mat_list]
    sub_mat_arr = np.vstack(sub_mat_list)
    # for large amount of data
    batch_size = 5000
    preds_l = []
    # i = 0
    # while i < len(sub_mat_arr):
    for i in tqdm(range(0, len(sub_mat_arr), batch_size)):
        preds_l.append(make_prediction(model, one_hot_mat=sub_mat_arr[i:min(i+batch_size, len(sub_mat_arr))]))
        if verbose:
            print(f"Predicted {round(i/len(sub_mat_arr), 2)}%")
        i += batch_size
    sub_seq_preds = np.vstack(preds_l)
    sub_seq_preds = sub_seq_preds.reshape(len(sub_seq_preds))
    seq_preds = [sub_seq_preds[preds_per_seq[i]:preds_per_seq[i+1]] for i in range(len(preds_per_seq)-1)]
    assert len(seq_preds) == len(seqs), f"ERROR: make_all_seqs_prediction - len(preds) != len(seq)"
    return [max(p) for p in seq_preds] if max_pred else seq_preds


def bar_plot(data_list):
    """
    Plots positions' predictions per sequence
    :param data_list: a list of predictions arrays
    """
    for data in data_list:
        x = [n for n in range(len(data))]
        plt.figure(figsize=(6, 4))
        plt.bar(x, data, width=1)
        plt.ylim([0, 1.2])
        plt.xlabel("Position")
        plt.ylabel("Prediction")
        plt.title("Transcript RSR Ratio Prediction")
        plt.show()


def plot_scores(scores_dict, y, dest):
    """
    Plots ROC/PR curves for a given predictions

    :param scores_dict: Dictionary of AUC_Score objects
    :param y: 1D array of labels For random guess calculation
    :param dest: str -Plot save destination(Optional)

    """
    legend_list = []
    scores_dict["G4Hunter"] = scores_dict["G4H"]
    scores_dict["cGcC-scoring"] = scores_dict["cGcC"]
    del scores_dict["cGcC"]
    del scores_dict["G4H"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in scores_dict:
        plt.plot(scores_dict[method].recall[1:], scores_dict[method].precision[1:])
        legend_list.append(f"{method} - {round(scores_dict[method].auc, 2)}")
    baseline = sum(y) / len(y)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    legend_list.append(f"Baseline = {round(baseline, 3)}")
    plt.legend(legend_list)
    plt.title(f"Human AUPR")
    plt.grid(alpha=0.3)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.savefig(f"{dest}/Human_AUCPR")




















