import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_input_size, one_hot_enc, pred_all_sub_seq, get_score_per_position
from Bio import SeqIO
import csv
import argparse
import matplotlib.pyplot as plt
from PARAMETERS import *

def bar_plot(data, desc, dst):
    x = [n for n in range(len(data))]
    plt.figure(figsize=(6, 4))
    plt.bar(x, data, width=1)
    plt.xlabel("Position")
    plt.ylabel("Prediction")
    plt.title(desc)
    plt.savefig(dst + f"/{desc}_detection")
    plt.show()


def predict_fasta(model, src, dst):
    # get sequences
    fasta_file = SeqIO.parse(open(src), 'fasta')

    seqs = [str(seq.seq).upper() for seq in fasta_file]
    print(f"Number of sequences = {len(seqs)}")

    # convert to one-hot-encoding and predict
    one_hot_mat_list = [one_hot_enc(x, False) for x in seqs]
    preds = [max(pred_all_sub_seq(x, model=model, pad=True)) for x in one_hot_mat_list]

    # save to file
    seqs_description = [s.description for s in SeqIO.parse(open(src), 'fasta')]
    preds_df = pd.DataFrame(data=list(zip(seqs_description, preds)), columns=["description", "rG4detector prediction"])
    preds_df.to_csv(dst + "/rG4detector_prediction.csv", index=False)


def detect_fasta(model, src, dst, plot):
    seqs_preds = []
    for io_seq in SeqIO.parse(open(src), 'fasta'):
        seq = str(io_seq.seq).upper()
        one_hot_mat = one_hot_enc(str(seq), False)
        preds = pred_all_sub_seq(one_hot_mat, model, pad=True)
        positions_score = get_score_per_position(preds, get_input_size(model), DETECTION_SIGMA)
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

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", dest="fasta_path", help="Fasta file for path", required=True)
    parser.add_argument("-m", "--model", dest="model_path", default="model/",
                        help="rG4detector model directory (default = model/)")
    parser.add_argument("-d", "--detect", action="store_true",
                        help="Operate in detection mode (default is evaluation mode)")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot detection results (for detection only)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-s", "--size", dest="ensemble_size", default=5, type=int, help="ensemble size (default 5)")
    args = parser.parse_args()

    rG4detector_model = []
    for i in range(args.ensemble_size):
        rG4detector_model.append(load_model(args.model_path + f"/model_{i}.h5"))
    if not args.detect:
        predict_fasta(model=rG4detector_model, src=args.fasta_path, dst=args.output)
    else:
        detect_fasta(model=rG4detector_model, src=args.fasta_path, dst=args.output, plot=args.plot)



