import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_input_size, get_score_per_position, make_all_seqs_prediction
from Bio import SeqIO
import csv
import argparse
import matplotlib.pyplot as plt
from PARAMETERS import *
from time import time
"""
This module is used to utilize the rG4detector model to perform predictions on a given fasta file.
rG4detector predictions can be utilized in two different modes:
1. prediction - for a given fasta file, rG4detector will predict the propensity of each sequence to form an rG4 
structure
2. detection - for a given fasta file, rG4detector will assign a score to each sequence nucleotide, which indicates the 
probability of this nucleotide to belong to an rG4 folding sequence.
"""

def bar_plot(data, desc, dst):
    """
    Plots positions' predictions per sequence

    :param data: 1D-array - predictions
    :param desc: str - sequence fasta description
    :param dst: str - path to destination directory
    """
    x = [n for n in range(len(data))]
    plt.figure(figsize=(6, 4))
    plt.bar(x, data, width=1)
    plt.xlabel("Position")
    plt.ylabel("Prediction")
    plt.title(desc)
    plt.savefig(dst + f"/{desc}_detection")
    plt.show()


def predict_fasta(model, src, dst):
    """
    Makes predictions for a given fasta file

    :param model: model object or list of models
    :param src: str - path to source fasta file
    :param dst: str - path to destination file to save results
    """
    # get sequences
    fasta_file = SeqIO.parse(open(src), 'fasta')
    seqs = [str(seq.seq).upper() for seq in fasta_file]
    print(f"Number of sequences = {len(seqs)}")

    # make predictions
    preds = make_all_seqs_prediction(model, seqs=seqs, pad="Z")

    # save to file
    seqs_description = [s.description for s in SeqIO.parse(open(src), 'fasta')]
    preds_df = pd.DataFrame(data=list(zip(seqs_description, preds)), columns=["description", "rG4detector prediction"])
    preds_df.to_csv(dst + "/rG4detector_prediction.csv", index=False)


def detect_fasta(model, src, dst, plot):
    """

    :param model: model object or list of models
    :param src: str - path to source fasta file
    :param dst: str - path to destination file to save results
    :param plot: boolean - plot predictions
    """
    # get sequences
    fasta_file = SeqIO.parse(open(src), 'fasta')
    seqs = [str(seq.seq).upper() for seq in fasta_file]
    print(f"Number of sequences = {len(seqs)}")

    # make predictions
    preds = make_all_seqs_prediction(model, seqs=seqs, max_pred=False, pad="Z", verbose=1)
    seqs_preds = []
    for p, io_seq in zip(preds, SeqIO.parse(open(src), 'fasta')):
        positions_score = get_score_per_position(p, get_input_size(model), DETECTION_SIGMA)
        seqs_preds.append(positions_score)
        if plot:
            bar_plot(positions_score, dst=dst, desc=io_seq.description)

    with open(dst + '/detection.csv', 'w') as f:
        write = csv.writer(f)
        for pred, io_seq in zip(seqs_preds, SeqIO.parse(open(src), 'fasta')):
            write.writerow([io_seq.description] + [n for n in str(io_seq.seq).upper()])
            write.writerow([""] + list(pred))
    return


if __name__ == "__main__":
    t = time()
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", dest="fasta_path", help="Fasta file for path", required=True)
    parser.add_argument("-m", "--model", dest="model_path", default="../model/",
                        help="rG4detector model directory (default = model/)")
    parser.add_argument("-d", "--detect", action="store_true",
                        help="Operate in detection mode (default is evaluation mode)")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot detection results (for detection only)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-s", "--size", dest="ensemble_size", default=11, type=int, help="ensemble size (default 11)")
    args = parser.parse_args()

    rG4detector_model = []
    for i in range(args.ensemble_size):
        rG4detector_model.append(load_model(args.model_path + f"/model_{i}.h5"))
    if not args.detect:
        predict_fasta(model=rG4detector_model, src=args.fasta_path, dst=args.output)
    else:
        detect_fasta(model=rG4detector_model, src=args.fasta_path, dst=args.output, plot=args.plot)

    if time() - t < 60:
        print(f"Execution time = {round(time()-t)}s")
    else:
        print(f"Execution time = {round((time() - t)/60, 1)}m")



