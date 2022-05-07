import sys
import time
from tensorflow.keras.models import load_model
import pickle
from seeker2transcript import get_transcript_dict
from Bio import SeqIO
import os.path
import getopt
from datetime import datetime
from detection_utils import *
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from utils import get_input_size

# Using only needed memory on GPU
con = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
con.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=con))

# rg4detector and screener detector
# both for Bartel and Kwok
# set rg4detector to True to recompute detection for rg4detector
# same for screener
# for Bartel detection set BARTEL to True
# for Kwok set BARTEL to False


DEBUG = False
BARTEL = False
data_set = "kwok"
rg4detector = False
screener = False
PLOT = False
CPU = False
TRAIN = False
canonical = False
not_canonical = False

debug_size = 1
gencode_path = f"big_files/gencode.v29lift37.transcripts.fa"
match_dict_path = 'detection/rg4-seeker-transcript-match-test.pkl'
model_dir = "models/best_model/ensemble/"
ensemble_size = 5
rg4_type = ""

opts, args = getopt.getopt(sys.argv[1:], 'psgbdm:o:i:cea:lPtny')
for op, val in opts:
    if op == "-d":
        DEBUG = True
    if op == "-p":
        PLOT = True
    if op == "-P":
        KPDS = True
    if op == "-g":
        rg4detector = True
    if op == "-b":
        BARTEL = True
        data_set = "bartel"
        match_dict_path = "detection/bartel/bartel-transcript-match.pkl"
    if op == "-c":
        CPU = True
    if op == "-s":
        screener = True
    if op == "-i":
        model_dir = val
    if op == "-t":
        TRAIN = True
        match_dict_path = 'detection/rg4-seeker-transcript-match-train.pkl'
        sigma = range(1, 20)
    if op == "-y":
        canonical = True
        rg4_type = "_canonical"
    if op == "-n":
        not_canonical = True
        rg4_type = "_not_canonical"


if canonical and BARTEL:
    match_dict_path = "detection/bartel/bartel-transcript-match-canonical.pkl"
if not_canonical and BARTEL:
    match_dict_path = "detection/bartel/bartel-transcript-match-not-canonical.pkl"
if canonical and not BARTEL:
    match_dict_path = f"detection/rg4-seeker-transcript-match-canonical.pkl"
if not_canonical and not BARTEL:
    match_dict_path = f"detection/rg4-seeker-transcript-match-not-canonical.pkl"


if CPU:
    print("Not using GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
detection_dir = model_dir + "/detection/"
output = detection_dir + data_set

if path.exists(output) is False:
    os.makedirs(output)

screener_path = "detection/screener/" + data_set

if not TRAIN:
    with open(detection_dir + f"/best_sigma.pkl", 'rb') as f:
        sigma = [pickle.load(f)]

def plot_scores(scores_dict, y, tag=rg4_type):
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M")
    transcriptom = "Human" if data_set == "kwok" else "Mouse"
    dest = output + f"/plots/"
    if path.exists(dest) is False:
        os.makedirs(dest)
    legend_list = []
    for method in scores_dict:
        plt.plot(scores_dict[method].recall[1:], scores_dict[method].precision[1:])
        legend_list.append(f"{method} - {round(scores_dict[method].auc, 3)}")
    baseline = sum(y) / len(y)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    legend_list.append(f"Baseline = {round(baseline, 3)}")
    plt.legend(legend_list)
    plt.title(f"{transcriptom} PR-AUC")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(dest + f"{data_set}_AUCPR{rg4_type}_{dt_string}")
    if PLOT:
        plt.show()


def main(model):
    t1 = time.time()
    # get input dim
    input_length = get_input_size(model)

    # get ground truth
    with open(match_dict_path, 'rb') as fp:
        exp_rg4 = pickle.load(fp)

    # get transcripts for rg4detector
    if rg4detector:
        if BARTEL:
            all_transcripts_dict = SeqIO.to_dict(SeqIO.parse("big_files/GCF_000001635_new.26_GRCm38.p6_rna.fna", "fasta"))
        else:
            all_transcripts_dict = get_transcript_dict(gencode_path)
        # keep only relevant transcripts
        transcript_dict = {}
        for transcript in exp_rg4:
            transcript_dict[transcript] = all_transcripts_dict[transcript]
        del all_transcripts_dict

    counter = 0
    print(f"number of transcripts = {len(exp_rg4)}")
    rg4detector_all_preds = None
    screener_all_preds = None
    t2 = time.time()
    for transcript in exp_rg4:
        counter += 1
        if counter % 100 == 0 or DEBUG:
            print(f"counter = {counter}, time = {round(time.time() - t2)}s")
            t2 = time.time()

        if rg4detector:
            seq = transcript_dict[transcript].seq
            one_hot_mat = one_hot_enc(str(seq))
            one_hot_mat = np.vstack((np.zeros((input_length-1, 4)), one_hot_mat, np.zeros((input_length-1, 4))))
            preds = pred_all_sub_seq(one_hot_mat, model)
            positions_score = get_score_per_position(preds, input_length, sigma)
            rg4detector_all_preds = positions_score if rg4detector_all_preds is None else \
                np.hstack((rg4detector_all_preds, positions_score))

        if screener:
            with open(screener_path + "/transcripts_score/" + transcript, 'rb') as fp:
                screener_scores = pickle.load(fp)
                screener_positions_score = set_screener_positions_scores(screener_scores)
            if screener_all_preds is None:
                screener_all_preds = screener_positions_score
            else:
                for method in screener_positions_score:
                    screener_all_preds[method] = np.vstack((screener_all_preds[method], screener_positions_score[method]))

        if DEBUG and counter == debug_size:
            if rg4detector:
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
    if rg4detector:
        # calc and save the best sigma if train is True
        if TRAIN:
            max_aupr = 0
            best_sigma = None
            for s in range(len(sigma)):
                precision, recall, t = precision_recall_curve(rg4_all_exp_seq, rg4detector_all_preds[s].
                                                              reshape(len(rg4detector_all_preds[s]), ))
                aupr = auc(recall, precision)
                f1 = 2/(1/recall + 1/precision)
                best_idx = f1.argmax()
                print(f"sigma = {sigma[s]}, auc = {aupr}")
                print(f"best threshold = {t[best_idx]}, f1 = {f1[best_idx]}")
                if aupr > max_aupr:
                    max_aupr = aupr
                    best_sigma = sigma[s]
            print(f"best sigma = {best_sigma}")
            if not DEBUG:
                with open(detection_dir + f"/best_sigma.pkl", "wb") as fp:
                    pickle.dump(best_sigma, fp)

            exit(0)
        # if not train calc performance
        precision, recall, t = precision_recall_curve(rg4_all_exp_seq,
                                                      rg4detector_all_preds.reshape(len(rg4detector_all_preds[0]), ))
        aupr = auc(recall, precision)
        if len(precision) > 1000000:
            precision, recall, t = precision[::10], recall[::10], t[::10]
        scores["rG4detector"] = PRScore("rg4detector", precision, recall, t, aupr)
        if not DEBUG:
            with open(output + f"/rg4detector_aucpr_score_all{rg4_type}.pkl", "wb") as fp:
                pickle.dump(scores["rG4detector"], fp)
    else:
        print("getting rg4detector score")
        with open(output + f"/rg4detector_aucpr_score_all{rg4_type}.pkl", 'rb') as fp:
            scores["rG4detector"] = pickle.load(fp)

    print(f"rg4detector scores:")
    print(scores["rG4detector"].auc)

    # plt.plot(recall, precision)
    # plt.show()

    # get screener score
    if screener:
        for method in ["G4NN", "cGcC", "G4H"]:
            precision, recall, t = precision_recall_curve(rg4_all_exp_seq, screener_all_preds[method])
            aupr = auc(recall, precision)
            if len(precision) > 1000000:
                precision, recall, t = precision[::10], recall[::10], t[::10]
            scores[method] = PRScore(method, precision, recall, t, aupr)
            with open(screener_path + f"/{method}_aucpr_score_all{rg4_type}.pkl", "wb") as fp:
                pickle.dump(scores[method], fp)
    else:
        print("getting screener score")
        for method in ["G4NN", "cGcC", "G4H"]:
            with open(screener_path + f"/{method}_aucpr_score_all{rg4_type}.pkl", 'rb') as fp:
                scores[method] = pickle.load(fp)
    for method in ["G4NN", "cGcC", "G4H"]:
        print(f"{method} scores:")
        print(scores[method].auc)

    print(f"exe time = {round((time.time()-t1)/60, 2)} minutes")
    plot_scores(scores, rg4_all_exp_seq)


if __name__ == "__main__":
    print(f"dir_path is {model_dir}")
    print(f"rg4detector is {rg4detector}")
    print(f"screener is {screener}")
    print(f"detection_dir is {detection_dir}")
    print(f"DEBUG is {DEBUG}")
    print(f"PLOT is {PLOT}")
    print(f"TRAIN is {TRAIN}")
    print(f"BARTEL is {BARTEL}")
    print(f"data_set is {data_set}")
    print(f"output = {output}")
    print(f"dict path = {match_dict_path}")


    if path.exists(output) is False:
        os.makedirs(output)
    if TRAIN and (BARTEL or screener):
        print("ERROR: BATEL or screener and TRAIN is True")
        exit(0)
    MODEL = []
    for i in range(ensemble_size):
        MODEL.append(load_model(model_dir + f"/model_{i}.h5"))

    main(MODEL)

















