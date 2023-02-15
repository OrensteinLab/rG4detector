from hyper_params import get_hyper_params, HyperParams
from tensorflow.keras.callbacks import EarlyStopping
import time
import getopt
import sys
from utils import get_data, set_data_size
import numpy as np
import random
import tensorflow as tf
from scipy.stats import pearsonr
from get_cnn_model import get_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PARAMETERS import *

def evaluate_model(x_train, y_train, x_val, y_val, hyper_params=HyperParams()):
    """

    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param hyper_params:
    :return:
    """
    print("Starting to evaluate model!")
    print(f"SEED = {hyper_params.seed}")
    # get model
    model = get_model(hyper_params)

    # fit model
    es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(x_train, y_train, verbose=0, validation_data=(x_val, y_val), epochs=hyper_params.epochs,
              batch_size=hyper_params.batch_size, callbacks=[es_callback])
    # make a prediction on the val set
    y_hat = model(x_val).numpy()
    y_hat = y_hat.reshape(len(y_hat))
    y_val = y_val.reshape(len(y_val))
    # evaluate model performance
    pr_corr = pearsonr(y_hat, y_val)[0]
    print(f"val corr = {pr_corr}")
    return pr_corr, model


def find_ensemble_size(x, y, dest, models_num, debug):
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


def main(hyper_params, model_num, iterations, dst, debug=False):
    start_time = time.time()
    [x_train, y_train, _], _, [x_val, y_val, _] = get_data(DATA_PATH, min_read=2000)
    if debug:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
    [x_train, x_val] = set_data_size(hyper_params.input_size, [x_train, x_val])

    corr_list = []
    seed_list = []
    for i in range(iterations):
        it_time = time.time()
        print(f"iteration: {i}/{iterations}")
        hyper_params.seed = random.randint(1, 1000)
        pr_corr, model = evaluate_model(x_train, y_train, x_val, y_val, hyper_params)
        corr_list.append(pr_corr)
        seed_list.append(hyper_params.seed)
        print("Finished Level - execution time = %ss ---\n\n" % (round(time.time() - it_time)))

    # reproduce best models:
    max_args = np.argsort(np.array(corr_list))[-model_num:][::-1]
    print(f"Reproducing {model_num} creating best models")
    for idx, loc in enumerate(max_args):
        it_time = time.time()
        print(f"SEED {idx} = {seed_list[loc]}")
        hyper_params.seed = seed_list[loc]
        _, model = evaluate_model(x_train, y_train, x_val, y_val, hyper_params)
        model.save(f"{dst}/model_{idx}.h5")
        print("Finished training - execution time = %ss ---\n\n" % (round(time.time() - it_time)))

    find_ensemble_size(x_val, y_val, dst, model_num, debug)
    print("Code execution time = %sm ---" % (round(time.time() - start_time) // 60))


if __name__ == "__main__":
    tf.random.set_seed(1)
    random.seed(1)
    np.random.seed(1)

    verb = 0
    num_of_iterations = NUM_OF_ENSEMBLE_ITERATIONS
    num_of_models = 15
    DEBUG = False
    output = MODEL_PATH

    opts, args = getopt.getopt(sys.argv[1:], 'o:d')
    for op, val in opts:
        if op == "-d":
            verb = 1
            epochs = 1
            DEBUG = True
            num_of_models = 1
            num_of_iterations = 1
        if op == "-o":
            output = val

    print(f"DEBUG is {DEBUG}")
    print(f"num_of_models is {num_of_models}")
    print(f"num_of_iterations = {num_of_iterations}")
    print(f"output is {output}")

    hyperParams = get_hyper_params(df_path=PARAMS_SCAN_PATH)
    main(hyperParams, num_of_models, num_of_iterations, output, debug=DEBUG)


