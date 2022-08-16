from calculate_correlation import calculate_human_correlation, calculate_mouse_correlation
from PARAMETERS import *

human_corr = calculate_human_correlation(MODEL_PATH, DATA_PATH)
mouse_corr = calculate_mouse_correlation(MODEL_PATH, DATA_PATH)

