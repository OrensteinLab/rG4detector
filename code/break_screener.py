import pandas as pd
import time
import pickle
from PARAMETERS import *

screener_output_file = SCREENER_DETECTION_PREDICTION_PATH + "seeker_detection_output.csv"
output_dir = SCREENER_DETECTION_PREDICTION_PATH


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
    with open(output_dir + "/" + transcript, "wb") as fp:
        pickle.dump(score_dict, fp)
    if counter % 100 == 0:
        print(f"counter = {counter}, time = {round(time.time()-t1)}s ")
        t1 = time.time()

print(f"time = {round(time.time()-t)/60,2}m minutes")

