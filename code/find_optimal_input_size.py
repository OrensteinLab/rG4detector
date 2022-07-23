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
from sklearn.metrics import mean_squared_error



tf.random.set_seed(1)
random.seed(10)
np.random.seed(1)


def evaluate_model(x_train, y_train, x_val, y_val, hyper_params=HyperParams()):
    print("Starting to evaluate model!")
    print(f"SEED = {hyper_params.seed}")
    # get model
    model = get_model(hyper_params)
    es_callback = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
    # fit model
    model.fit(x_train, y_train, verbose=verb, validation_data=(x_val, y_val),
              batch_size=hyper_params.batch_size, callbacks=[es_callback])
    # make a prediction on the val set
    y_hat = model(x_val).numpy()
    y_hat = y_hat.reshape(len(y_hat))
    y_val = y_val.reshape(len(y_val))
    # evaluate model performance
    pr_corr = pearsonr(y_hat, y_val)[0]
    mse = mean_squared_error(y_val, y_hat)
    print(f"mse = {mse}")
    print(f"val corr = {pr_corr}")
    return pr_corr, mse


def plot_scores(corr_list, rng, debug):
    # plot results
    plt.plot(rng, corr_list)
    plt.xlabel("Input size")
    plt.ylabel("Score")
    plt.title("Performance vs input size")
    plt.savefig(f"../docs/input_size")
    if debug:
        plt.show()


def main(hyper_params, iterations, debug):
    [x_train, y_train, _], _, [x_val, y_val, _] = get_data(DATA_PATH, min_read=2000)
    if DEBUG:
        x_train = x_train[:1000]
        y_train = y_train[:1000]

    scores = []
    mse_scores = []
    rng = range(70, 160, 10) if not debug else range(70, 90, 10)
    for s in rng:
        print(f"size = {s}")
        hyper_params.input_size = s
        [x_train_i, x_val_i] = set_data_size(s, [x_train, x_val])
        corr_list = []
        mse_list = []
        it_time = time.time()
        for i in range(iterations):
            print(f"iteration: {i}/{iterations}")
            hyper_params.seed = random.randint(1, 1000)
            pr_corr, mse = evaluate_model(x_train_i, y_train, x_val_i, y_val, hyper_params)
            corr_list.append(pr_corr)
            mse_list.append(mse)
        print(f"correlation = {sum(corr_list)/len(corr_list)} +-{np.std(corr_list)}")
        print(f"mse = {sum(mse_list)/len(mse_list)} +-{np.std(mse_list)}")
        scores.append(sum(corr_list)/len(corr_list))
        mse_scores.append(sum(mse_list)/len(mse_list))
        print(f"Finished {s}- execution time = %ss ---\n\n" % (round(time.time() - it_time)))

    plot_scores(scores, rng, debug)
    plot_scores(mse_scores, rng, debug)



if __name__ == "__main__":
    tf.random.set_seed(1)
    random.seed(10)
    np.random.seed(1)

    verb = 1
    DEBUG = False
    num_of_iterations = 5 if not DEBUG else 1


    print(f"DEBUG is {DEBUG}")
    print(f"num_of_iterations = {num_of_iterations}")

    hyperParams = get_hyper_params(df_path=PARAMS_SCAN_PATH)
    main(hyperParams, num_of_iterations, DEBUG)


