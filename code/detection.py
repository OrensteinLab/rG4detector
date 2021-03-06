import sys
import time
from tensorflow.keras.models import load_model
import pickle
import getopt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from utils import *
import matplotlib.pyplot as plt
from seeker2transcript import get_transcript_dict
from PARAMETERS import *
import numpy as np

DEBUG = False

def bar_plot(data_list):
    for data in data_list:
        x = [n for n in range(len(data))]
        plt.figure(figsize=(6, 4))
        plt.bar(x, data, width=1)
        plt.ylim([0, 1.2])
        plt.xlabel("Position")
        plt.ylabel("Prediction")
        # plt.title("Positive rG4 location")
        plt.title("Transcript RSR Ratio Prediction")
        plt.show()

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
    plt.show()
    plt.savefig(f"../detection/results/Human_AUCPR")


def detect_rg4(model):
    t1 = time.time()
    # get input dim
    input_length = get_input_size(model)
    # get ground truth
    with open(DETECTION_RG4_SEEKER_HITS, 'rb') as fp:
        exp_rg4 = pickle.load(fp)
    # get transcripts for rg4detector
    all_transcripts_dict = get_transcript_dict(HUMAN_V29_TRANSCRIPTOME_PATH)
    # keep only relevant transcripts
    transcript_dict = {}
    for transcript in exp_rg4:
        transcript_dict[transcript] = all_transcripts_dict[transcript]
    del all_transcripts_dict
    print(f"Number of transcripts = {len(exp_rg4)}")
    counter = 0
    # get screener predictions
    with open(SCREENER_DETECTION_PREDICTION_PATH, 'rb') as fp:
        screener_scores = pickle.load(fp)

    # predict all transcripts
    rg4detector_all_preds = None
    screener_all_preds = None
    t2 = time.time()
    for transcript in exp_rg4:
        counter += 1
        if counter % 100 == 0 or DEBUG:
            print(f"counter = {counter}, time = {round(time.time() - t2)}s")
            t2 = time.time()
        # make rG4detector prediction
        seq = transcript_dict[transcript].seq
        one_hot_mat = one_hot_enc(str(seq))
        # zero padding
        one_hot_mat = np.vstack((np.zeros((input_length-1, 4)), one_hot_mat, np.zeros((input_length-1, 4))))
        preds = pred_all_sub_seq(one_hot_mat, model)
        positions_score = get_score_per_position(preds, input_length, DETECTION_SIGMA)

        # s = 0
        # e = 0
        # pos = []
        # while s < len(exp_rg4[transcript]):
        #     while s < len(exp_rg4[transcript]) and not exp_rg4[transcript][s] :
        #         s += 1
        #     if s == len(exp_rg4[transcript]):
        #         break
        #     e = s
        #     while e < len(exp_rg4[transcript]) and exp_rg4[transcript][e]:
        #         e += 1
        #
        #     pos.append((s, e))
        #     s = e

        # for s, e in pos:
        #     bar_plot([exp_rg4[transcript][s-100:e+100], positions_score[s-100:e+100]])
        rg4detector_all_preds = positions_score if rg4detector_all_preds is None else \
            np.hstack((rg4detector_all_preds, positions_score))

        # if counter == 5:
        #     exit(0)
        # continue

        # get screener predictions
        screener_positions_score = set_screener_positions_scores(screener_scores[transcript])
        if screener_all_preds is None:
            screener_all_preds = screener_positions_score
        else:
            for method in screener_positions_score:
                screener_all_preds[method] = np.vstack((screener_all_preds[method], screener_positions_score[method]))

        if DEBUG and counter == 2:
            del transcript_dict
            break

    # stack all ground truth data
    counter = 0
    rg4_all_exp_seq = None
    for transcript in exp_rg4:
        rg4_all_exp_seq = exp_rg4[transcript] if rg4_all_exp_seq is None else np.hstack((rg4_all_exp_seq,
                                                                                         exp_rg4[transcript]))
        counter += 1
        if DEBUG and counter == 2:
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

    print(f"Execution time = {round((time.time()-t1)/60, 2)} minutes")
    # TODO - remove
    plot_scores(scores, rg4_all_exp_seq)

    # save data
    for m in scores:
        with open(f"../detection/results/{m}_detection_aupr.csv", "w") as f:
            f.write(f"precision,recall\n")
            if len(scores[m].precision) > 1000000:
                scores[m].precision, scores[m].recall = scores[m].precision[::10], scores[m].recall[::10]
            for precision, recall in zip(scores[m].precision, scores[m].recall):

                f.write(f"{precision},{recall}\n")



if __name__ == "__main__":
    print(f"Starting detection")
    MODEL = []
    for i in range(ENSEMBLE_SIZE):
        MODEL.append(load_model(MODEL_PATH + f"/model_{i}.h5"))
    detect_rg4(MODEL)

















