import sys
from PARAMETERS import *
from utils import get_data
import numpy as np

if len(sys.argv) < 3:
    print("Execution: create_screener_fasta.py <data_dir_path> <output_file_name> [OPTIONAL: <window_size>]")
    exit(0)
data_dir_path = sys.argv[1]
output_dir = sys.argv[2]
input_size = 60
if len(sys.argv) == 4:
    input_size = int(sys.argv[3])

# Read test sequences
_, [x_test, y_test, _], _ = get_data(DATA_PATH)
nuc_idx = np.argmax(x_test, axis=2)
x_arr_str = np.char.mod('%d', nuc_idx)
x_str_list = ["".join(x) for x in x_arr_str]
sequences = [x.translate(x.maketrans('0123', 'ACGT')) for x in x_str_list]
chop_size = (len(sequences[0])-input_size)//2
sequences_center = [s[chop_size:chop_size+input_size] for s in sequences]

# Write sequences centers to fasta file
with open(output_dir, 'w') as f:
    for i, s in enumerate(sequences_center):
        f.write(f">{i}\n{s}\n")
