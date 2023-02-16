from hyper_params import get_hyper_params, HyperParams
from tensorflow.keras.callbacks import EarlyStopping
import time
from utils import get_data, set_data_size
import numpy as np
import random
import tensorflow as tf
from scipy.stats import pearsonr
from get_cnn_model import get_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PARAMETERS import *
import argparse
from tqdm import tqdm
from os import environ

def evaluate_model(x_train, y_train, x_val, y_val, hyper_params=HyperParams(), verbose=0):
    """
    This function gets data and hyper-params as input and returns a trained model and its performance

    :param x_train: Train data - array-like of shape (n_samples, n_features)
    :param y_train: Data labels - 1D array-like
    :param x_val: Validation data - array-like of shape (n_samples, n_features)
    :param y_val: Validation labels - 1D array-like
    :param hyper_params: Model hyper parameters - HyperParams object
    :param verbose: Model training verbosity - int(2/1/0)
    :return: (pr_corr, model) - (model pearson correlation on the validation set, model object)
    """

    if verbose:
        print("Starting to evaluate model!")
        print(f"SEED = {hyper_params.seed}")
    # get model
    model = get_model(hyper_params)

    # fit model
    es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=verbose)
    model.fit(x_train, y_train, verbose=0, validation_data=(x_val, y_val), epochs=hyper_params.epochs,
              batch_size=hyper_params.batch_size, callbacks=[es_callback])
    # make a prediction on the val set
    y_hat = model(x_val).numpy()
    y_hat = y_hat.reshape(len(y_hat))
    y_val = y_val.reshape(len(y_val))
    # evaluate model performance
    pr_corr = pearsonr(y_hat, y_val)[0]
    if verbose:
        print(f"val corr = {pr_corr}")
    return pr_corr, model


def find_ensemble_size(x, y, dest, models_num, debug):
    """
    This function plots the model performance as a function of the ensemble size

    :param x: Test data - array-like of shape (n_samples, n_features)
    :param y: Data labels - 1D array-like
    :param dest: path to model directory - str
    :param models_num: largest ensemble size to rest
    :param debug: True for debug
    """
    y = y.reshape(len(y))
    ensemble_preds = np.zeros((len(y), models_num))
    for i in range(models_num):
        # get ith model preds
        model = load_model(f"{dest}/model_{i}.h5")
        y_hat = model(x).numpy()
        pred = y_hat.reshape(len(y_hat))
        # add pred to the relevant bagging size
        for j in range(models_num, i, -1):
            ensemble_preds[:, j-1] += pred

    # calc pearson correlation
    pr_list = []
    for i in range(models_num):
        pr_list.append(pearsonr(ensemble_preds[:, i], y)[0])

    # plot results
    plt.plot(range(1, models_num + 1), pr_list)
    plt.xlabel("Number of Models")
    plt.ylabel("Pearson Correlation")
    plt.title("Correlation vs Ensemble Size")
    plt.savefig(f"{dest}/ensemble_size")
    if debug:
        plt.show()


def main(hyper_params, model_num, iterations, dst, debug=False, verbose=0):
    start_time = time.time()
    [x_train, y_train, _], _, [x_val, y_val, _] = get_data(DATA_PATH, min_read=2000)
    if debug:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
    [x_train, x_val] = set_data_size(hyper_params.input_size, [x_train, x_val])

    corr_list = []
    seed_list = []
    print(f"Starting to train {iterations} models")
    for _ in tqdm(range(iterations)):
        hyper_params.seed = random.randint(1, 1000)
        pr_corr, model = evaluate_model(x_train, y_train, x_val, y_val, hyper_params, verbose=verbose)
        corr_list.append(pr_corr)
        seed_list.append(hyper_params.seed)

    # reproduce best models:
    max_args = np.argsort(np.array(corr_list))[-model_num:][::-1]
    print(f"Reproducing {model_num} best models")
    for idx, loc in enumerate(max_args):
        hyper_params.seed = seed_list[loc]
        _, model = evaluate_model(x_train, y_train, x_val, y_val, hyper_params)
        model.save(f"{dst}/model_{idx}.h5")

    find_ensemble_size(x_val, y_val, dst, model_num, debug)
    print("Code execution time = %sm ---" % (round(time.time() - start_time) // 60))


if __name__ == "__main__":
    tf.random.set_seed(1)
    random.seed(1)
    np.random.seed(1)

    verb = 0
    num_of_iterations = NUM_OF_ENSEMBLE_ITERATIONS
    num_of_models = 15

    # TODO - REMOVE
    environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory", default=MODEL_PATH)
    parser.add_argument("-d", "--debug", dest="DEBUG", help="Define in debug mode", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", help=f"Model training verbosity", type=int, default=0)
    args = parser.parse_args()

    if args.DEBUG:
        args.verbose = 1
        epochs = 1
        num_of_models = 1
        num_of_iterations = 1


    print(f"DEBUG is {args.DEBUG}")
    print(f"num_of_models is {num_of_models}")
    print(f"num_of_iterations = {num_of_iterations}")
    print(f"output is {args.output}\n")

    hyperParams = get_hyper_params(df_path=PARAMS_SCAN_PATH)
    main(hyperParams, num_of_models, num_of_iterations, args.output, debug=args.DEBUG, verbose=args.verbose)


