import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PARAMETERS import *
from utils import one_hot_enc, set_data_size
import matplotlib.pyplot as plt
import pandas as pd
import logomaker

model = load_model(MODEL_PATH + f"/model_0.h5")

def linearly_interpolate(sample, num_steps=50):
    reference = np.ones(sample.shape)*0.25

    # Calculated stepwise difference from reference to the actual sample.
    ret = np.zeros(([num_steps] + [i for i in sample.shape]))
    for s in range(num_steps):
        ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

    return ret  # num_steps, (sample - reference) * (1.0 / num_steps)


def get_gradients(sample):

    with tf.GradientTape() as tape:
        tape.watch(sample)
        preds = model(sample)
        top_class = preds[:, 0]

    grads = tape.gradient(top_class, sample)
    return grads


def get_integrated_gradients(sample, num_steps=50):

    baseline = np.ones(sample.shape)*0.25

    # 1. Do interpolation.
    interpolated_sample = linearly_interpolate(sample)
    # interpolated_sample = np.array(interpolated_sample).astype(np.float32)

    # 3. Get the gradients
    grads = []
    for i, s in enumerate(interpolated_sample):
        s = tf.expand_dims(s, axis=0)
        grad = get_gradients(s)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (sample - baseline) * avg_grads
    return integrated_grads






