# rG4detector
We present rG4detector, a convolutional neural network for predicting rG4 folding of any given sequence based on rG4-seq data. rG4-seq  experimental protocol was employed to generate transcriptome-wide rG4 maps of human HeLa cells (accession code GSE77282). In this study, we used the experimental raw data to extract RSR (ratio of stalled reads) scores across the transcriptome, and used this information to train rG4detector. 
rG4detector assigns an rG4 propensity score for any RNA sequence. in addition, we utilized rG4detector predictions to  detect rG4 forming sequences in a given transcript/any long RNA sequence.

## Get rG4detector predictions
rG4detector can operate in two different modes:
  1. prediction - for a given fasta file, rG4detector will predict the propensity of each sequence to form an rG4 structure.
  2. detection - for a given fasta file, rG4detector will assign a score to each sequence nucleotide, which indicates the probability of this nucleotide to belong to an rG4 folding sequence.

### Prediction mode
In order to predict the propensity of each sequence in a fasta file to form an rG4 structure, run:
```
python code/predict_fasta.py -f <fasta_file> -o <output_directory>
```
If the rG4detector models aren't under ```models/```, pass the model directory using -m flag.
The program will output the results to the output directory under ```rG4detector_prediction.csv```.

### Detection mode
In order to utilize rG4detector predictions in a detection task use the -d/--detect flags, as follow:
```
python code/predict_fasta.py -d -f <fasta_file> -o <output_directory>
```
The program will output the results to the output directory under ```rG4detector_detection.csv```.
For plotting each sequence scores, use the -p/--plot flags (recommended for fasta files with limited number of sequences).

## Reproduce rG4detector model

### Prerequisites

The model is implemented with Keras python library with Tensorflow backend (version 2.5.0).

### Model training

```
python code/train_rG4detector.py 
```
The model will be saved under ```model/``` .
