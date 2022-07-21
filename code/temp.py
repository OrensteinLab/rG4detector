from hyper_params import HyperParams
import os
import time
from tensorflow.keras.models import load_model
import pickle
import os.path
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from seeker2transcript import get_transcript_dict
from utils import *
import random
from train_rG4detector import evaluate_model
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


tf.random.set_seed(1)
random.seed(10)
np.random.seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DEBUG = False
debug_size = 1


def set_seq(seq, data_size, return_pad_size=False, extra=0):
    if len(seq) > data_size:
        start = len(seq) // 2 - data_size // 2
        end = start + data_size
        return seq[start:end]
    else:
        trailing_padding = data_size//2 - 15 - extra
        leading_padding = data_size - len(seq) - trailing_padding
        if return_pad_size:
            return "N" * leading_padding + seq + "N" * trailing_padding, leading_padding, trailing_padding
        else:
            return "N" * leading_padding + seq + "N" * trailing_padding

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

    tmp = max(pr_list)
    return pr_list.index(tmp) + 1


def detect_rg4(model, input_length):
    t1 = time.time()
    # get ground truth
    with open(DETECTION_RG4_SEEKER_HITS, 'rb') as fp:
        exp_rg4 = pickle.load(fp)
    # get transcripts for rg4detector
    all_transcripts_dict = get_transcript_dict(HUMAN_TRANSCRIPTOME_PATH)
    # keep only relevant transcripts
    transcript_dict = {}
    for transcript in exp_rg4:
        transcript_dict[transcript] = all_transcripts_dict[transcript]
    del all_transcripts_dict
    print(f"Number of transcripts = {len(exp_rg4)}")
    counter = 0
    rg4detector_all_preds = None
    t2 = time.time()
    # predict all transcripts
    for transcript in exp_rg4:
        counter += 1
        if counter % 100 == 0 or DEBUG:
            print(f"counter = {counter}, time = {round(time.time() - t2)}s")
            t2 = time.time()
        seq = transcript_dict[transcript].seq
        one_hot_mat = one_hot_enc(str(seq), remove_last=False)
        # zero padding
        one_hot_mat = np.vstack((np.zeros((input_length-1, 4)), one_hot_mat, np.zeros((input_length-1, 4))))
        preds = pred_all_sub_seq(one_hot_mat, model)
        positions_score = get_score_per_position(preds, input_length, DETECTION_SIGMA)
        rg4detector_all_preds = positions_score if rg4detector_all_preds is None else \
            np.hstack((rg4detector_all_preds, positions_score))

        if DEBUG and counter == debug_size:
            del transcript_dict
            break

    # stack all ground truth data
    counter = 0
    rg4_all_exp_seq = None
    for transcript in exp_rg4:
        rg4_all_exp_seq = exp_rg4[transcript] if rg4_all_exp_seq is None else np.hstack((rg4_all_exp_seq,
                                                                                         exp_rg4[transcript]))
        counter += 1
        if DEBUG and counter == debug_size:
            break
    del exp_rg4

    # calc rg4detector score
    precision, recall, t = precision_recall_curve(rg4_all_exp_seq,
                                                  rg4detector_all_preds.reshape(len(rg4detector_all_preds), ))
    return auc(recall, precision)


def get_hyper_params(df_path, num):
    hyper_params_l = []
    hyper_params = HyperParams()
    print(f"getting best hyper-parameters")
    pref_df = pd.read_csv(df_path)
    pref_df = pref_df.sort_values(by=["val_corr"], ascending=False, ignore_index=True)
    class_methods = vars(hyper_params)
    for i in range(num):
        hyper_params = HyperParams()
        best_row = pref_df.iloc[i]
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
        hyper_params_l.append(hyper_params)
    return hyper_params_l


def human_correlation(model):
    _, [xTest, yTest, _], _ = get_data(DATA_PATH, min_read=2000)
    data_size = get_input_size(model)
    [xTest] = set_data_size(data_size, [xTest])
    preds = np.zeros((len(xTest), 1))
    for j in range(len(model)):
        preds += model[j](xTest).numpy() / len(model)
    preds = preds.reshape(len(preds))
    yTest = yTest.reshape(len(yTest))
    corr = pearsonr(preds, yTest)[0]
    return corr


def check_mouse_correlation(model, input_length):
    print("Computing mouse correlation")
    # params
    mouse_df = pd.read_csv("../data/mouse/mouse_data.csv", names=["sequence", "label"], header=None, delimiter="\t")
    chop_size = (len(mouse_df.loc[0, "sequence"]) - input_length) // 2
    sequences = [s[chop_size:chop_size + input_length + 1] for s in mouse_df["sequence"]]
    X = np.array(list(map(one_hot_enc, sequences)))

    pred = np.zeros((len(X), 1))
    for m in model:
        pred += m(X).numpy()/len(model)
    pred = pred.reshape(len(pred))
    log_rsr = np.log(mouse_df["label"])
    corr = pearsonr(pred, log_rsr)
    return round(corr[0], 3)


def main(hyper_params, model_num, iterations, dst, debug=False):
    start_time = time.time()
    [x_train, y_train, _], _, [x_val, y_val, _] = get_data(DATA_PATH, min_read=2000)
    if debug:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
    [x_train, x_val] = set_data_size(hyper_params.input_size, [x_train, x_val])


    for i in range(iterations):
        it_time = time.time()
        print(f"iteration: {i}/{iterations}")
        hyper_params.seed = random.randint(1, 1000)
        pr_corr, model = evaluate_model(x_train, y_train, x_val, y_val, hyper_params)
        model.save(f"{dst}/model_{i}.h5")
        print("Finished Level - execution time = %ss ---\n\n" % (round(time.time() - it_time)))


    return find_ensemble_size(x_val, y_val, dst, model_num, debug)


def loop_length_test2(model, data_size, output=None, plot=True):
    data = pd.read_csv("../interpretation/amy_data.csv", index_col="loop")
    data["Delta Gvh"] = -data["Delta Gvh"]

    for idx in data.index:
        loop_length_l = list(idx)
        loops = []
        for loop_len in loop_length_l[1:]:
            if int(loop_len) == 1:
                loops.append("H")
            else:
                loops.append("H" + "N" * (int(loop_len)-2) + "H")
        seq = "HGGG" + loops[0] + "GGG" + loops[1] + "GGG" + loops[2] + "GGGH"
        data.loc[idx, "rG4detector"] = make_prediction(model, set_seq(seq, data_size, extra=1))[0][0]

    sp_coef, sp_p = spearmanr(data["Delta Gvh"], data["rG4detector"])

    reg = LinearRegression().fit(data["Delta Gvh"].to_numpy().reshape(-1, 1), data["rG4detector"].to_numpy().reshape(-1, 1))
    reg_coef = reg.coef_[0][0]
    ref_intercept = reg.intercept_[0]

    x_range = [x for x in np.arange(3, 35, 3)]

    ax = data.plot.scatter(x="Delta Gvh", y="rG4detector")
    ax.plot([x_range[0], x_range[-1]], [x_range[0]*reg_coef + ref_intercept, x_range[-1]*reg_coef + ref_intercept],
             '--', c="0.5")

    plt.xlabel(r'-$\Delta$G$_V$$_H$')
    # plt.xticks(x_range)
    plt.legend([f"Spearman coefficient = {round(sp_coef, 3)}"])
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for loop, location in data.iterrows():
        # plt.text(location["Delta Gvh"], location["rG4detector"], loop[1:], c="b")
        if loop[1:] == "131":
            plt.annotate(loop[1:], (location["Delta Gvh"]-1.5, location["rG4detector"]+0.02))
        else:
            plt.annotate(loop[1:], location)
    if output:
        plt.savefig(output + f"/Loop test 2")
    if plot:
        plt.show()
    return sp_coef

def test_models():
    output_path = "../temp/scores.csv"
    scores = []
    hyper_params_l = get_hyper_params(PARAMS_SCAN_PATH, 20)
    for i, hyper_params in enumerate(hyper_params_l):
        model_path = f"../temp/model_{i}/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        ens_size = main(hyper_params, 7, 7, model_path, debug=DEBUG)
        models = []
        for j in range(ens_size):
            models.append(load_model(model_path + f"model_{i}.h5"))
        human_corr = human_correlation(models)
        print(f'Human correlation = {round(human_corr, 3)}')
        mouse_corr = check_mouse_correlation(models, hyper_params.input_size)
        print(f'Mouse correlation = {round(mouse_corr, 3)}')
        sp_coef = loop_length_test2(model=models, data_size=hyper_params.input_size, plot=False)
        print(f'sp_coef correlation = {round(sp_coef, 3)}')
        detect = detect_rg4(models, hyper_params.input_size)
        print(f'Detection aupr = {round(detect, 3)}')
        scores.append((human_corr, mouse_corr, detect, sp_coef))


        with open(output_path, 'w') as f:
            f.write(f'Human\tMouse\tdetect\tsp_coef\n')
            for s in scores:
                f.write(f'{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\n')


if __name__ == "__main__":
    test_models()






