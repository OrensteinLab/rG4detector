from hyper_params import HyperParams
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Input, concatenate
from tensorflow.keras.regularizers import l2


def conv_block(layer_input, conv_idx, hyper_params=HyperParams()):
    x = Conv1D(filters=hyper_params.conv_size[conv_idx], kernel_size=hyper_params.kernel_size[conv_idx],
               input_shape=(hyper_params.input_size, 4),
               kernel_regularizer=l2(hyper_params.l2_regularization[conv_idx]))(layer_input)
    x = MaxPool1D(pool_size=hyper_params.pool_size[conv_idx])(x)
    x = Dropout(hyper_params.dropout[conv_idx])(x)
    return Flatten()(x)

def get_model(hyper_params=HyperParams()):
    tf.random.set_seed(hyper_params.seed)
    input_layer = Input(shape=(hyper_params.input_size, 4))
    # conv blocks
    conv_block_list = []
    for i in range(hyper_params.kernel_num):
        conv_block_list.append(conv_block(layer_input=input_layer, conv_idx=i, hyper_params=hyper_params))
    # concat conv blocks output
    x = concatenate(conv_block_list) if len(conv_block_list) > 1 else conv_block_list[0]
    # add fc layers
    for i in range(hyper_params.dense_num):
        x = Dense(hyper_params.dense_size[0], activation='relu')(x)
        x = Dropout(hyper_params.dropout[i+3])(x)
    # prediction neuron
    output_layer = Dense(1, activation='linear')(x)

    model = Model(input_layer, output_layer)
    opt = tf.keras.optimizers.Adam(learning_rate=hyper_params.lr)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model
