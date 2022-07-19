import sys

if len(sys.argv) < 3:
    print("Execution: create_screener_fasta.py <path_to_test_data> <output_file_name> [OPTIONAL: <window_size>]")
    exit(0)
data_file_path = sys.argv[1]
output_dir = sys.argv[2]
input_size = 60
if len(sys.argv) == 4:
    input_size = int(sys.argv[3])

# Read test sequences
with open(data_file_path, "r") as f:
    sequences = f.read().splitlines()
chop_size = (len(sequences[0])-input_size)//2
sequences_center = [s[chop_size:chop_size+input_size] for s in sequences]

# Write sequences centers to fasta file
with open(output_dir, 'w') as f:
    for i, s in enumerate(sequences_center):
        f.write(f">{i}\n{s}\n")
