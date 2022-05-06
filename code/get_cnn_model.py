from hyper_params import HyperParams
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Input, concatenate
from tensorflow.keras.regularizers import l2


def get_model(hyper_params=HyperParams()):
    print("Multiple kernel CNN model")
    tf.random.set_seed(hyper_params.seed)
    input_layer = Input(shape=(hyper_params.input_size, 4))

    conv_layer_list = []
    # convolution layer # 1
    conv_layer_1 = Conv1D(filters=hyper_params.conv_size_1, kernel_size=hyper_params.multi_kernel_1,
                          input_shape=(hyper_params.input_size, 4), name="conv_1",
                          kernel_regularizer=l2(hyper_params.l2_regularization_1))(input_layer)
    conv_layer_1 = MaxPool1D(pool_size=hyper_params.multi_pool_size_1, name="pooling_1")(conv_layer_1)
    conv_layer_1 = Dropout(hyper_params.dropout_1)(conv_layer_1)
    conv_layer_1 = Flatten()(conv_layer_1)
    conv_layer_list.append(conv_layer_1)

    # convolution layer # 2
    if hyper_params.kernel_num > 1:
        conv_layer_2 = Conv1D(filters=hyper_params.conv_size_2, kernel_size=hyper_params.multi_kernel_2,
                              input_shape=(hyper_params.input_size, 4), name="conv_2", padding="same",
                              kernel_regularizer=l2(hyper_params.l2_regularization_2))(input_layer)
        conv_layer_2 = MaxPool1D(pool_size=hyper_params.multi_pool_size_2, name="pooling_2")(conv_layer_2)
        conv_layer_2 = Dropout(hyper_params.dropout_1)(conv_layer_2)
        conv_layer_2 = Flatten()(conv_layer_2)
        conv_layer_list.append(conv_layer_2)

    if hyper_params.kernel_num == 3:
        conv_layer_3 = Conv1D(filters=hyper_params.conv_size_3, kernel_size=hyper_params.multi_kernel_3,
                              input_shape=(hyper_params.input_size, 4), name="conv_3", padding="same",
                              kernel_regularizer=l2(hyper_params.l2_regularization_3))(input_layer)
        conv_layer_3 = MaxPool1D(pool_size=hyper_params.multi_pool_size_3, name="pooling_3")(conv_layer_3)
        conv_layer_3 = Dropout(hyper_params.dropout_1)(conv_layer_3)
        conv_layer_3 = Flatten()(conv_layer_3)
        conv_layer_list.append(conv_layer_3)

    if hyper_params.kernel_num > 1:
        next_layer = concatenate(conv_layer_list)
    else:
        next_layer = conv_layer_1

    model = Dense(hyper_params.dense_size_1, activation='relu', name="dense")(next_layer)
    model = Dropout(hyper_params.dropout_2)(model)
    model = Dense(hyper_params.dense_size_2, activation='relu', name="dense2")(model)
    model = Dense(1, activation='linear', name="1dense")(model)
    model = Model(input_layer, model)
    opt = tf.keras.optimizers.Adam(learning_rate=hyper_params.lr)
    model.compile(loss='mean_squared_error', optimizer='adam')
    # plot_model(model)
    return model