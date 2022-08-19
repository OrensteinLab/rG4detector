# import tensorflow as tf
# from keras.models import Model, Sequential
# from tensorflow.keras.models import load_model
# from PARAMETERS import *
# import numpy as np
# import keras.backend as K
# from utils import one_hot_enc, set_data_size
#
#
# model = load_model(MODEL_PATH + f"/model_0.h5")
# input_tensors = []
# for i in model.inputs:
#     input_tensors.append(i)
# input_tensors.append(K.learning_phase())
# a = model.output
# b = model.input
# gradients = model.optimizer.get_gradients(model.output[:], model.input)
# get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#
# class integrated_gradients:
#
#     def __init__(self, model):
#         self.model = model
#
#
#     @staticmethod
#     def linearly_interpolate(sample, num_steps=50):
#         reference = np.ones(sample.shape)*0.25
#
#         # Calculated stepwise difference from reference to the actual sample.
#         ret = np.zeros(([num_steps] + [i for i in sample.shape]))
#         for s in range(num_steps):
#             ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)
#
#         return ret, num_steps, (sample - reference) * (1.0 / num_steps)
#
#     def explain(sample, num_steps=50):
#
#         # Each element for each input stream.
#         samples = []
#         numsteps = []
#         step_sizes = []
#
#         # If multiple inputs are present, feed them as list of np arrays.
#         if isinstance(sample, list):
#             # If reference is present, reference and sample size need to be equal.
#             for s in range(len(sample)):
#                 _output = linearly_interpolate(sample[s], num_steps)
#                 samples.append(_output[0])
#                 numsteps.append(_output[1])
#                 step_sizes.append(_output[2])
#
#         # Or you can feed just a single numpy arrray.
#         elif isinstance(sample, np.ndarray):
#             _output = linearly_interpolate(sample, num_steps)
#             samples.append(_output[0])
#             numsteps.append(_output[1])
#             step_sizes.append(_output[2])
#
#         # For tensorflow backend
#         _input = []
#         for s in samples:
#             _input.append(s)
#         _input.append(0)
#
#         grad = get_gradients(_input)
#
#         explanation = []
#         for g in range(len(grad)):
#             _temp = np.sum(grad[g], axis=0)
#             explanation.append(np.multiply(_temp, step_sizes[g]))
#
#         # Format the return values according to the input sample.
#         if isinstance(sample, list):
#             return explanation
#         elif isinstance(sample, np.ndarray):
#             return explanation[0]
#         return -1
#
#
# seq = "CTGCTGCCGCTACTGCGGAGTAGCTGCTTCCCTTCCTCCTCTCCCGGCGGCGGCGGCGGCAGCGGCGGAGGAGGAGGAGGAGGGGACCCGGGCGCAGAGAGCCG" \
#       "GCCGGCGGCGCAGTTGCAGCGCGGAG"
# hot_mat = np.array(one_hot_enc(seq)).reshape((1, len(seq), 4))
# hot_mat = set_data_size(80, [hot_mat])[0]
# a = explain(hot_mat)
# b = 5
#
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






