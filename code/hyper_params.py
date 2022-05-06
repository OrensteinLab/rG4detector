import random

class HyperParams:
    def __init__(self):
        self.input_size_list = [x for x in range(70, 150, 10)]
        self.conv_size_list = [16, 32, 64, 128]
        self.dense_size_list = [16, 32, 64, 128]
        self.dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lr_list = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.rnn_size_list = [10, 20, 30]
        self.batch_size_list = [16, 32, 64, 128]
        self.epochs_list = [x for x in range(1, 10)]
        self.regularization_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        self.kernel_num = random.choice([1, 2, 3])
        self.input_size = random.choice(self.input_size_list)
        self.conv_size = random.choice(self.conv_size_list)
        self.dense_size_1 = random.choice(self.dense_size_list)
        self.dense_size_2 = random.choice(self.dense_size_list)
        self.dropout_1 = random.choice(self.dropout_list)
        self.dropout_2 = random.choice(self.dropout_list)
        self.dropout_3 = random.choice(self.dropout_list)
        self.lr = random.choice(self.lr_list)
        self.rnn_size = random.choice(self.rnn_size_list)
        self.batch_size = random.choice(self.batch_size_list)
        self.kernel_size = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_1 = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_2 = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_3 = random.choice([x for x in range(2, self.input_size + 1)])
        self.pool_size = random.choice([x for x in range(1, self.input_size - self.kernel_size + 2)])
        self.epochs = random.choice(self.epochs_list)
        self.bid_rnn = random.choice([True, False])
        self.second_dense = random.choice([True, False])
        self.conv_size_1 = random.choice(self.conv_size_list)
        self.conv_size_2 = random.choice(self.conv_size_list)
        self.conv_size_3 = random.choice(self.conv_size_list)
        self.multi_pool_size_1 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_1 + 2)])
        self.multi_pool_size_2 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_2 + 2)])
        self.multi_pool_size_3 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_3 + 2)])
        self.l1_regularization = random.choice(self.regularization_list)
        self.l2_regularization_1 = random.choice(self.regularization_list)
        self.l2_regularization_2 = random.choice(self.regularization_list)
        self.l2_regularization_3 = random.choice(self.regularization_list)
        self.seed = random.randint(1, 10000)


    def rand_params(self):
        self.input_size = random.choice(self.input_size_list)
        self.conv_size = random.choice(self.conv_size_list)
        self.dense_size_1 = random.choice(self.dense_size_list)
        self.dense_size_2 = random.choice(self.dense_size_list)
        self.dropout_1 = random.choice(self.dropout_list)
        self.dropout_2 = random.choice(self.dropout_list)
        self.dropout_3 = random.choice(self.dropout_list)
        self.lr = random.choice(self.lr_list)
        self.rnn_size = random.choice(self.rnn_size_list)
        self.batch_size = random.choice(self.batch_size_list)
        self.kernel_size = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_1 = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_2 = random.choice([x for x in range(2, self.input_size + 1)])
        self.multi_kernel_3 = random.choice([x for x in range(2, self.input_size + 1)])
        self.pool_size = random.choice([x for x in range(1, self.input_size - self.kernel_size + 2)])
        self.epochs = random.choice(self.epochs_list)
        self.bid_rnn = random.choice([True, False])
        self.second_dense = random.choice([True, False])
        self.conv_size_1 = random.choice(self.conv_size_list)
        self.conv_size_2 = random.choice(self.conv_size_list)
        self.conv_size_3 = random.choice(self.conv_size_list)
        self.multi_pool_size_1 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_1 + 2)])
        self.multi_pool_size_2 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_2 + 2)])
        self.multi_pool_size_3 = random.choice([x for x in range(1, self.input_size - self.multi_kernel_3 + 2)])
        self.l1_regularization = random.choice(self.regularization_list)
        self.l2_regularization_1 = random.choice(self.regularization_list)
        self.l2_regularization_2 = random.choice(self.regularization_list)
        self.l2_regularization_3 = random.choice(self.regularization_list)
        self.seed = random.randint(1, 10000)


def get_hyper_params(best=False):
    hyper_params = HyperParams()
    if best:
        hyper_params.kernel_num = 3
        hyper_params.input_size = 130
        hyper_params.dense_size_1 = 128
        hyper_params.dense_size_2 = 32
        hyper_params.lr = 0.0001
        hyper_params.batch_size = 128
        hyper_params.multi_kernel_1 = 42
        hyper_params.multi_kernel_2 = 74
        hyper_params.multi_kernel_3 = 23
        hyper_params.epochs = 5
        hyper_params.conv_size_1 = 32
        hyper_params.conv_size_2 = 16
        hyper_params.conv_size_3 = 32
        hyper_params.l2_regularization_1 = 0.005
        hyper_params.l2_regularization_2 = 0.01
        hyper_params.multi_pool_size_1 = 50
        hyper_params.multi_pool_size_2 = 39
        hyper_params.multi_pool_size_3 = 3
        hyper_params.dropout_1 = 0.5
        hyper_params.dropout_2 = 0.1
        hyper_params.l2_regularization_1 = 0.1
        hyper_params.l2_regularization_2 = 0.001
        hyper_params.l2_regularization_3 = 0.005
    return hyper_params

