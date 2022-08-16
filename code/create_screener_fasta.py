from utils import get_data
import argparse
import pandas as pd


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", dest="data_dir_path", help="data directory path", required=True)
parser.add_argument("-o", "--output", dest="output_file", help="Output file for screener fasta file", required=True)
parser.add_argument("-s", "--size", dest="data_size", default=60, help="screener data size", type=int)
parser.add_argument("-m", "--mouse", action="store_true", help="create mouse data")
parser.add_argument("-f", "--full", dest="full_size", help="creates same data as rG4detector", type=int, default=None)
args = parser.parse_args()

if not args.mouse:
    print("Creating human fasta file for screener")
    # Read test sequences
    _, [sequences, _, _], _ = get_data(args.data_dir_path, get_seq=True)
    if args.full_size:
        start = len(sequences[0]) // 2 - args.full_size // 2
        end = start + args.full_size
    else:
        end = len(sequences[0])//2 + 15 + 5  # evaluated RTS cite location + 5 for confidence
        start = end - args.data_size
    sequences_center = [s[start:end] for s in sequences]

    # Write sequences centers to fasta file
    with open(args.output_file, 'w') as f:
        for i, s in enumerate(sequences_center):
            f.write(f">{i}\n{s}\n")
    print("Done creating human fasta file for screener")


else:
    print("Creating mouse fasta file for screener")
    # Read test sequences and labels
    data_df = pd.read_csv(args.data_dir_path + f"mouse_data.csv", names=["sequence", "labels"], header=None)
    if args.full_size:
        start = len(data_df.loc[0, "sequence"]) // 2 - args.full_size // 2
        end = start + args.full_size
    else:
        end = len(data_df.loc[0, "sequence"])//2 + 15 + 2  # evaluated RTS site location + 2 for confidence
        start = end - args.data_size
    data_df["sequence"] = data_df["sequence"].map(lambda x: x[start:end])
    with open(args.output_file, 'w') as f:
        for idx, row in data_df.iterrows():
            f.write(f'>{idx}\n{row["sequence"]}\n')
    print("Done creating mouse fasta file for screener")



