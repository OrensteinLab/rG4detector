import pandas as pd
import time
import pickle
import sys
import getopt

window_length = 29
step_size = 1
screener_output_file = None
output_file = None
if len(sys.argv) < 3:
    print("ERROR:\nExecution: break_screener.py -i <screener_output_csv> -o <output_file>"
          " [options: -s <step_size(def=1)> -w <window_size(def=29)>]")
    exit(0)


opts, args = getopt.getopt(sys.argv[1:], 'i:o:d:w:s:')
for op, val in opts:
    if op == "-i":
        screener_output_file = val
    if op == "-o":
        output_file = val
    if op == "-s":
        step_size = int(val)
    if op == "-w":
        window_length = int(val)


t = time.time()
screener_output = pd.read_csv(screener_output_file, delimiter="\t")
transcripts_names = screener_output["description"].unique()
transcripts_dict = {}
print(f"Number of transcripts = {len(transcripts_names)}")
counter = 0
t1 = time.time()
for transcript in transcripts_names:
    score_dict = {}
    counter += 1
    trans_df = screener_output[screener_output["description"] == transcript]
    score_dict["cGcC"] = trans_df["cGcC"].to_numpy()
    score_dict["G4H"] = trans_df["G4H"].to_numpy()
    score_dict["G4NN"] = trans_df["G4NN"].to_numpy()
    transcripts_dict[transcript] = score_dict
    if counter % 100 == 0:
        print(f"counter = {counter}, time = {time.time()-t1} ")
        t1 = time.time()
print(f"Number of transcript: {len(transcripts_dict)}")
with open(output_file, "wb") as fp:  # Pickling
    pickle.dump(transcripts_dict, fp)

print(f"Done! execution time = {round(time.time()-t/60,2)} minutes")

