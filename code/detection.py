import sys
import time
from tensorflow.keras.models import load_model
import pickle
from Bio import SeqIO
import os.path
import getopt
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from utils import *
import matplotlib.pyplot as plt
from seeker2transcript import get_transcript_dict
from PARAMETERS import *
import numpy as np


DEBUG = False
PLOT = False
debug_size = 1


def plot_scores(scores_dict, y):
    dest = f"detection/"
    legend_list = []
    for method in scores_dict:
        plt.plot(scores_dict[method].recall[1:], scores_dict[method].precision[1:])
        legend_list.append(f"{method} - {round(scores_dict[method].auc, 3)}")
    baseline = sum(y) / len(y)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    legend_list.append(f"Baseline = {round(baseline, 3)}")
    plt.legend(legend_list)
    plt.title(f"Human AUPR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(dest + f"Human_AUCPR")


def detect_rg4(model):
    t1 = time.time()
    # get input dim
    input_length = get_input_size(model)
    # get ground truth
    with open(DETECTION_RG4_SEEKER_HITS, 'rb') as fp:
        exp_rg4 = pickle.load(fp)
    # get transcripts for rg4detector
    all_transcripts_dict = get_transcript_dict(HUMAN_TRANSCRIPTOME_PATH)
    # keep only relevant transcripts
    transcript_dict = {}
    for transcript in exp_rg4:
        transcript_dict[transcript] = all_transcripts_dict[transcript]
    del all_transcripts_dict
    print(f"Number of transcripts = {len(exp_rg4)}")
    counter = 0

    rg4detector_all_preds = None
    screener_all_preds = None
    t2 = time.time()
    # predict all transcripts
    for transcript in exp_rg4:
        counter += 1
        if counter % 100 == 0 or DEBUG:
            print(f"counter = {counter}, time = {round(time.time() - t2)}s")
            t2 = time.time()

        seq = transcript_dict[transcript].seq
        one_hot_mat = one_hot_enc(str(seq), remove_last=False)
        # zero padding
        one_hot_mat = np.vstack((np.zeros((input_length-1, 4)), one_hot_mat, np.zeros((input_length-1, 4))))
        preds = pred_all_sub_seq(one_hot_mat, model)
        positions_score = get_score_per_position(preds, input_length, DETECTION_SIGMA)
        rg4detector_all_preds = positions_score if rg4detector_all_preds is None else \
            np.hstack((rg4detector_all_preds, positions_score))

        with open(SCREENER_DETECTION_PREDICTION_PATH + transcript, 'rb') as fp:
            screener_scores = pickle.load(fp)
        screener_positions_score = set_screener_positions_scores(screener_scores)
        if screener_all_preds is None:
            screener_all_preds = screener_positions_score
        else:
            for method in screener_positions_score:
                screener_all_preds[method] = np.vstack((screener_all_preds[method], screener_positions_score[method]))

        if DEBUG and counter == debug_size:
            del transcript_dict
            break

    # stack all ground truth data
    counter = 0
    rg4_all_exp_seq = None
    for transcript in exp_rg4:
        rg4_all_exp_seq = exp_rg4[transcript] if rg4_all_exp_seq is None else np.hstack((rg4_all_exp_seq,
                                                                                         exp_rg4[transcript]))
        counter += 1
        if DEBUG and counter == debug_size:
            break
    del exp_rg4

    scores = {}
    # calc rg4detector score
    precision, recall, t = precision_recall_curve(rg4_all_exp_seq,
                                                  rg4detector_all_preds.reshape(len(rg4detector_all_preds), ))
    aupr = auc(recall, precision)
    scores["rG4detector"] = PRScore("rg4detector", precision, recall, t, aupr)
    print(f"rG4detector score:")
    print(scores["rG4detector"].auc)


    # get screener score
    for method in METHODS_LIST:
        precision, recall, t = precision_recall_curve(rg4_all_exp_seq, screener_all_preds[method])
        aupr = auc(recall, precision)
        scores[method] = PRScore(method, precision, recall, t, aupr)

    for method in METHODS_LIST:
        print(f"{method} scores:")
        print(scores[method].auc)

    print(f"exe time = {round((time.time()-t1)/60, 2)} minutes")
    plot_scores(scores, rg4_all_exp_seq)


if __name__ == "__main__":
    print(f"Starting detection")
    opts, args = getopt.getopt(sys.argv[1:], 'dp')
    for op, val in opts:
        if op == "-d":
            DEBUG = True
        if op == "-p":
            PLOT = True

    MODEL = []
    for i in range(ENSEMBLE_SIZE):
        MODEL.append(load_model(MODEL_PATH + f"/model_{i}.h5"))

    detect_rg4(MODEL)

















