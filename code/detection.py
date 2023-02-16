import time
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from utils import *
from seeker2transcript import get_transcript_dict
from PARAMETERS import *
import numpy as np
import argparse


def detect_rg4(model, rg4_seeker_hits, gencode_path, screener_path, dest, screener_window):

    t1 = time.time()
    # get input dim
    input_length = get_input_size(model)
    # get ground truth
    with open(rg4_seeker_hits, 'rb') as fp:
        exp_rg4 = pickle.load(fp)

    # get transcripts for rg4detector
    all_transcripts_dict = get_transcript_dict(gencode_path)
    # keep only relevant transcripts
    transcript_dict = {}
    for s in exp_rg4:
        transcript_dict[s] = all_transcripts_dict[s]
    del all_transcripts_dict
    print(f"Number of transcripts = {len(exp_rg4)}")

    # get screener predictions
    with open(screener_path, 'rb') as fp:
        screener_scores = pickle.load(fp)

    preds = {}
    # make rG4detector prediction
    print("Detecting with rG4detector")
    seqs = ["Z"*(input_length-1) + str(transcript_dict[s].seq) + "Z"*(input_length-1) for s in exp_rg4]
    predictions = make_all_seqs_prediction(model=model, seqs=seqs, max_pred=False)
    preds["rG4detector"] = [get_score_per_position(p, input_length, DETECTION_SIGMA) for p in predictions]

    # get screener predictions
    print("Getting G4RNA screener predictions")
    screener_positions_scores = [set_screener_positions_scores(screener_scores[s], gaussian=True, average=True,
                                                               window_size=screener_window) for s in exp_rg4]
    for m in METHODS_LIST:
        preds[m] = [score[m] for score in screener_positions_scores]

    # get all ground truth data
    rg4_all_exp_seq = np.hstack([exp_rg4[s] for s in exp_rg4])

    scores = {}
    # calc aupr score
    print("Starting to calculate AUPR")
    for m in preds:
        print(f"Calculating {m} AUPR")
        if m != "G4H":
            precision, recall, t = precision_recall_curve(rg4_all_exp_seq, np.hstack(preds[m]))
        else:
            t = np.unique(np.hstack(preds[m]))[:-1] + 0.001
            precision, recall = get_G4Hunter_roc(sequences=transcript_dict,
                                                 predictions_l=preds[m],
                                                 thresholds=t,
                                                 ground_truth=exp_rg4,
                                                 screener_window=screener_window)
        aupr = auc(recall, precision)
        scores[m] = PRScore(m, precision, recall, t, aupr)
        print(f"{m} score: {scores[m].auc}")

    
    print(f"Execution time = {round((time.time()-t1)/60, 2)} minutes")



    # save data
    if dest:
        print("Plotting results")
        for m in scores:
            if len(scores[m].precision) > 1000000:
                scores[m].precision, scores[m].recall = scores[m].precision[::10], scores[m].recall[::10]

        plot_scores(scores, rg4_all_exp_seq, dest)
        print("Saving results")
        for m in scores:
            with open(dest + f"/{m}_detection_aupr.csv", "w") as f:
                f.write(f"precision,recall\n")

                for precision, recall in zip(scores[m].precision, scores[m].recall):
                    f.write(f"{precision},{recall}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_path", help="rG4detector model directory", required=True)
    parser.add_argument("-s", "--seeker", dest="rg4_seeker_hits",
                        help="rG4-seeker hits processed file path "
                             "(located under detection/rg4-seeker-transcript-match-test.pkl)", required=True)
    parser.add_argument("-i", "--input", dest="screener_input",
                        help="Screener predictions (located under g4rna_screener/detection/screener_predictions.pkl)",
                        required=True)
    parser.add_argument("-g", "--gencode", dest="gencode_path", help="Human gencode v40 file path", required=True)
    parser.add_argument("-e", "--ensemble", dest="ensemble_size",
                        help=f"rG4detector ensemble size (default={ENSEMBLE_SIZE})", default=ENSEMBLE_SIZE)
    parser.add_argument("-d", "--dest", dest="dest", help=f"Path for results", default=None)
    parser.add_argument("-w", "--window", dest="screener_window", help=f"G4RNA screener window size", type=int,
                        default=80)
    args = parser.parse_args()

    print(f"Starting detection")
    MODEL = []
    for i in range(args.ensemble_size):
        MODEL.append(load_model(args.model_path + f"/model_{i}.h5"))
    detect_rg4(
        model=MODEL,
        rg4_seeker_hits=args.rg4_seeker_hits,
        gencode_path=args.gencode_path,
        screener_path=args.screener_input,
        dest=args.dest,
        screener_window=args.screener_window)

















