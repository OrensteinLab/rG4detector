DATA_PATH = "../data/"
MODEL_PATH = "../model/"
SCREENER_PATH = "../screener/"
GENOMES_FILE = "../rg4detector/big_files/"
PARAMS_SCAN_PATH = "../docs/scan_performance.csv"
MOUSE_TRANSCRIPTOME_PATH = GENOMES_FILE + "GCF_000001635_new.26_GRCm38.p6_rna.fna"
HUMAN_TRANSCRIPTOME_PATH = GENOMES_FILE + "gencode.v29lift37.transcripts.fa"
REFERANCE_GENOME = GENOMES_FILE + "hg38.fa"

# GENOMES_FILE = "big_files/"
SCREENER_DETECTION_PATH = SCREENER_PATH + "detection/"
SCREENER_DETECTION_PREDICTION_PATH = SCREENER_DETECTION_PATH + "transcripts_predictions/"
ENSEMBLE_SIZE = 5
NUM_OF_ENSEMBLE_ITERATIONS = 50
MOUSE_PATH = "independent_datasets/mouse/"
MOUSE_DATA_PATH = MOUSE_PATH + "Table_S1_mouse.csv"

MOUSE_SCREENER_LENGTH = 60
DATA_SIZE = 130
DETECTION_SIGMA = 17
METHODS_LIST = ['cGcC', 'G4H', 'G4NN']
# DETECTION_SCREENER_LENGTH =
DETECTION_RG4_SEEKER_HITS = "detection/rg4-seeker-transcript-match.pkl"
RG4_SEEKER_PATH = "data/rg4-seq_rts.k.rG4_list.combined.csv"
G4RNA_SCREENER = "G4RNA/screener_preds.csv"

