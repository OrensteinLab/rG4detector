from utils import get_data
import numpy as np
import argparse
import pandas as pd


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", dest="data_dir_path", help="data directory path", required=True)
parser.add_argument("-o", "--output", dest="output_file", help="Output file for screener fasta file", required=True)
parser.add_argument("-s", "--size", dest="data_size", default=60, help="screener data size", type=int)
parser.add_argument("-m", "--mouse", action="store_true", help="create mouse data")
args = parser.parse_args()

if not args.mouse:
    print("Creating human fasta file for screener")
    # Read test sequences
    _, [x_test, y_test, _], _ = get_data(args.data_dir_path)
    nuc_idx = np.argmax(x_test, axis=2)
    x_arr_str = np.char.mod('%d', nuc_idx)
    x_str_list = ["".join(x) for x in x_arr_str]
    sequences = [x.translate(x.maketrans('0123', 'ACGT')) for x in x_str_list]
    chop_size = (len(sequences[0])-args.data_size)//2
    sequences_center = [s[chop_size:chop_size+args.data_size] for s in sequences]

    # Write sequences centers to fasta file
    with open(args.output_file, 'w') as f:
        for i, s in enumerate(sequences_center):
            f.write(f">{i}\n{s}\n")
    print("Done creating human fasta file for screener")


else:
    print("Creating mouse fasta file for screener")
    # Read test sequences and labels
    data_df = pd.read_csv(args.data_dir_path + f"mouse_data.csv", names=["sequence", "labels"], header=None)
    chop_size = (len(data_df.loc[0, "sequence"])-args.data_size)//2
    data_df["sequence"] = data_df["sequence"].map(lambda x: x[chop_size:chop_size+args.data_size])
    with open(args.output_file, 'w') as f:
        for idx, row in data_df.iterrows():
            f.write(f'>{idx}\n{row["sequence"]}\n')
    print("Done creating mouse fasta file for screener")



