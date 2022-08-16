import random
import pandas as pd

class HyperParams:
    def __init__(self):
        self.input_size_range = range(70, 150, 10)
        self.conv_size_range = [16, 32, 64, 128]
        self.dense_size_range = [16, 32, 64, 128]
        self.dropout_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lr_range = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.rnn_size_range = [10, 20, 30]
        self.batch_size_range = [16, 32, 64, 128]
        self.epochs_range = [x for x in range(1, 10)]
        self.input_size = random.choice(self.input_size_range)
        self.kernel_num = random.choice([1, 2, 3])
        self.dense_size = [random.choice(self.dense_size_range) for _ in range(3)]
        self.dropout = [random.choice(self.dropout_range) for _ in range(6)]
        self.lr = random.choice(self.lr_range)
        self.batch_size = random.choice(self.batch_size_range)
        self.kernel_size_range = [x for x in range(2, self.input_size + 1)]
        self.kernel_size = [random.choice(self.kernel_size_range) for _ in range(3)]
        self.pool_size_range = [[x for x in range(1, self.input_size - self.kernel_size[i] + 2)] for i in range(3)]
        self.pool_size = [random.choice(self.pool_size_range[i]) for i in range(3)]
        self.epochs = random.choice(self.epochs_range)
        self.dense_num = random.choice([1, 2])
        self.conv_size = [random.choice(self.conv_size_range) for _ in range(3)]
        self.l2_regularization = [random.choice(self.lr_range) for _ in range(3)]
        self.seed = random.randint(1, 10000)

    def rand_params(self):
        self.input_size = random.choice(self.input_size_range)
        self.kernel_num = random.choice([1, 2, 3])
        self.dense_size = [random.choice(self.dense_size_range) for _ in range(3)]
        self.dropout = [random.choice(self.dropout_range) for _ in range(6)]
        self.lr = random.choice(self.lr_range)
        self.batch_size = random.choice(self.batch_size_range)
        self.kernel_size_range = [x for x in range(2, self.input_size + 1)]
        self.kernel_size = [random.choice(self.kernel_size_range) for _ in range(3)]
        self.pool_size_range = [[x for x in range(1, self.input_size - self.kernel_size[i] + 2)] for i in range(3)]
        self.pool_size = [random.choice(self.pool_size_range[i]) for i in range(3)]
        self.epochs = random.choice(self.epochs_range)
        self.dense_num = random.choice(range(1,4))
        self.conv_size = [random.choice(self.conv_size_range) for _ in range(3)]
        self.l2_regularization = [random.choice(self.lr_range) for _ in range(3)]
        self.seed = random.randint(1, 10000)


def get_hyper_params(df_path=None):
    hyper_params = HyperParams()
    if df_path is not None:
        print(f"getting best hyper-parameters")
        pref_df = pd.read_csv(df_path)
        pref_df = pref_df.sort_values(by=["val_corr"], ascending=False)
        class_methods = vars(hyper_params)
        best_row = pref_df.iloc[0]
        for m in class_methods:
            if "range" not in m:
                if isinstance(class_methods[m], list):
                    if "l2" in m or "dropout" in m:
                        setattr(hyper_params, m, list(map(float, best_row[m][1:-1].split(","))))
                    else:
                        setattr(hyper_params, m, list(map(int, best_row[m][1:-1].split(","))))
                else:
                    if "lr" in m:
                        setattr(hyper_params, m, float(best_row[m]))
                    else:
                        setattr(hyper_params, m, int(best_row[m]))

                print(f"{m} = {best_row[m]}")
    return hyper_params



