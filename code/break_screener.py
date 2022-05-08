import pandas as pd
import time
import pickle
import sys
from PARAMETERS import *

window_length = 29
step_size = 1
data_set = "kwok"
screener_output_file = f"big_files/seeker_detection_score_{window_length}_{step_size}.csv"
output_dir = f"detection/screener/{data_set}/transcripts_score/{window_length}/"
if len(sys.argv) > 2:
    screener_output_file = sys.argv[1]
    output_dir = sys.argv[2]



t = time.time()
screener_output = pd.read_csv(screener_output_file, delimiter="\t")
transcripts_names = screener_output["description"].unique()
score_dict = {}
print(f"Number of transcripts = {len(transcripts_names)}")
counter = 0
t1 = time.time()
for transcript in transcripts_names:
    counter += 1
    trans_df = screener_output[screener_output["description"] == transcript]
    score_dict["cGcC"] = trans_df["cGcC"].to_numpy()
    score_dict["G4H"] = trans_df["G4H"].to_numpy()
    score_dict["G4NN"] = trans_df["G4NN"].to_numpy()
    with open(output_dir + "/" + transcript, "wb") as fp:  # Pickling
        pickle.dump(score_dict, fp)
    if counter % 100 == 0:
        print(f"counter = {counter}, time = {time.time()-t1} ")
        t1 = time.time()

print(f"time = {(time.time()-t)/60} minutes")

