from tensorflow.keras.models import load_model
import getopt
import sys
from scipy.stats import pearsonr, spearmanr
from utils import get_input_size, make_prediction, set_data_size, one_hot_enc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import logomaker
from PARAMETERS import *
from IG import get_integrated_gradients

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


def loop_length_test(model, data_size, output):
    max_loop_size = 12
    loop_predictions = {}
    data = {"loop_size": range(1, max_loop_size + 1)}
    fig, ax = plt.subplots()
    loops_idx = ["First", "Second", "Third"]
    for loop in range(3):
        loop_size_dict = {0: 1, 1: 1, 2: 1}
        seq_list = []
        for loop_size in range(1, max_loop_size + 1):
            loop_size_dict[loop] = loop_size
            seq = "GGG" + "N" * loop_size_dict[0] + "GGG" + "N" * loop_size_dict[1] + "GGG" + "N" * loop_size_dict[2] \
                  + "GGG"
            seq_list.append(set_seq(seq, data_size))
        loop_predictions[loop] = make_prediction(model, seq_list)
        data[f"loop_{loop+1}"] = loop_predictions[loop].reshape(max_loop_size,)
        ax.plot(range(1, max_loop_size + 1), loop_predictions[loop], 'o--', label=f"{loops_idx[loop]} loop")
    ax.set_xticks(range(1, max_loop_size+1))
    plt.xlabel("Loop length")
    plt.ylabel("rG4detector prediction")
    plt.xlim([0.5, max_loop_size + 0.5])
    # plt.title(output_path)
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.savefig(output + f"/Loop length influence", dpi=400)
    # TODO
    if PLOT:
        plt.show()
    data = pd.DataFrame.from_dict(data)
    data.to_csv(output + f"Loop test 1.csv", index=False)


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
    plt.legend([f"Spearman coefficient = {round(sp_coef, 2)}"])
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.text(location["Delta Gvh"], location["rG4detector"], loop[1:], c="b")
    # for loop, location in data.iterrows():
    #     if loop[1:] == "333":
    #         plt.annotate(loop[1:], (location["Delta Gvh"], location["rG4detector"]-0.05))
    #     elif loop[1:] == "223":
    #         plt.annotate(loop[1:], (location["Delta Gvh"], location["rG4detector"]-0.05))
    #     elif loop[1:] == "232":
    #         plt.annotate(loop[1:], (location["Delta Gvh"]-1.5, location["rG4detector"]))
    #     elif loop[1:] == "131":
    #         plt.annotate(loop[1:], (location["Delta Gvh"]-1.5, location["rG4detector"]))
    #     else:
    #         plt.annotate(loop[1:], location)
    if output:
        plt.savefig(output + f"/Loop length combination test", dpi=400)
    if plot:
        plt.show()
    data.to_csv(output + f"Loop test 2.csv")
    return sp_coef


def mutation_effect(model, data_size, output):
    pq_seq_list = list("GGGNGGGNGGGNGGG")
    df_idx = ["A", "C", "G", "T"]

    # add base pred to the end of the list
    base_pred = make_prediction(model, set_seq("".join(pq_seq_list), data_size))
    base_pred = base_pred.reshape(len(base_pred))

    preds = np.zeros((len(df_idx), len(pq_seq_list)))
    for base in range(len(df_idx)):
        for position in range(len(pq_seq_list)):
            if pq_seq_list[position] == "N" or base == 2:
                preds[base, position] = base_pred
                continue
            mutated_seq_list = pq_seq_list.copy()
            mutated_seq_list[position] = df_idx[base]
            seq = set_seq("".join(mutated_seq_list), data_size)
            if position == len(pq_seq_list) - 1:
                seq = "N" + seq[:-1]
            preds[base, position] = make_prediction(model, seq)

    preds_df = pd.DataFrame(data=preds-base_pred, columns=pq_seq_list, index=df_idx, dtype=float)

    # preds_df = preds_df - base_pred
    plt.pcolormesh(preds_df, cmap="RdBu")
    plt.colorbar()
    plt.xticks([x + 0.5 for x in range(len(preds_df.columns))], preds_df.columns)
    # plt.xticks()
    plt.yticks([x + 0.5 for x in range(len(preds_df.index))], preds_df.index)
    plt.title("Mutation Influence")
    plt.savefig(output + f"/mutation influence", dpi=400)
    if PLOT:
        plt.show()



def mutation_map_test(model, data_size, output, seq):
    hot_mat = np.array(one_hot_enc(seq)).reshape((1, len(seq), 4))
    hot_mat = set_data_size(data_size, [hot_mat])[0]
    pred = make_prediction(model, one_hot_mat=hot_mat)
    hot_mat = hot_mat.reshape(data_size, 4)
    logo_mats = []
    mutation_mats = []

    # create mutated sequences
    for p in range(data_size):
        temp_mat = hot_mat.copy()
        base = np.argmax(temp_mat[p, :])
        temp_mat[p, :] = [1/3, 1/3, 1/3, 1/3]
        temp_mat[p, base] = 0
        logo_mats.append(temp_mat)
        for n in range(4):
            temp_mat = hot_mat.copy()
            temp_mat[p, :] = np.eye(4)[n, :]
            mutation_mats.append(temp_mat)

    # predict sequences
    # logo
    logo_mats = np.array(logo_mats)
    logo_preds = pred - make_prediction(model, one_hot_mat=logo_mats)
    # mutation_map
    mutation_mats = np.array(mutation_mats)
    mutation_preds = make_prediction(model, one_hot_mat=mutation_mats) - pred
    # set preds to mats
    logo_mat = hot_mat.copy()
    mutation_map = np.zeros((4, data_size))
    for idx, p in enumerate(logo_preds):
        logo_mat[idx, :] *= p
        mutation_map[:, idx] = mutation_preds[idx*4:(idx+1)*4].reshape(4)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(25, 5))
    v_max = max(np.max(mutation_map), -np.min(mutation_map))
    v_min = - v_max
    pcm = ax2.pcolormesh(mutation_map, cmap="RdBu", vmax=v_max, vmin=v_min)
    ax2.set_yticks([x + 0.5 for x in range(4)])
    ax2.set_yticklabels(["A", "C", "G", "T"])
    fig.colorbar(pcm, ax=ax2, pad=.005, fraction=.01)
    plt.ylabel("Variant", fontsize=16, labelpad=17)
    plt.xlabel("Input position", fontsize=16)
    # create logo
    crp_logo = logomaker.Logo(pd.DataFrame(logo_mat, columns=["A", "C", "G", "T"]),
                              shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold', ax=ax1)
    # style using Logo methods
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    # style using Axes methods
    crp_logo.ax.set_ylabel("Attribution score", labelpad=-1, fontsize=14)
    crp_logo.ax.set_xticks([])
    plt.subplots_adjust(hspace=.001)
    logo_position = ax1.get_position()
    ax1.set_position([logo_position.x0, logo_position.y0, logo_position.x1*0.848, logo_position.y1*0.55])
    plt.savefig(output + f"sequence mutation map 2", dpi=400)
    # if PLOT:
    #     plt.show()


    # IG method
    # fig, (ax1, ax2) = plt.subplots(2, figsize=(25, 5))
    ax1.clear()
    igrad = get_integrated_gradients(hot_mat).numpy()
    logo_mat = igrad * hot_mat
    # fig, ax = plt.subplots(1, figsize=(25, 5))
    crp_logo = logomaker.Logo(pd.DataFrame(logo_mat, columns=["A", "C", "G", "T"]),
                              shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold', ax=ax1)
    # style using Logo methods
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    # style using Axes methods
    crp_logo.ax.set_ylabel("Attribution score", labelpad=-1, fontsize=14)
    crp_logo.ax.set_xticks([])
    plt.subplots_adjust(hspace=.001)
    logo_position = ax1.get_position()
    ax1.set_position([logo_position.x0, logo_position.y0, logo_position.x1 * 0.848, logo_position.y1 * 0.55])
    plt.savefig(output + f"IG_2", dpi=400)
    plt.show()


# def loc_check(model, data_size):
#     for j in range(1, 6):
#         print(j)
#         seq = "GGG" + "N"*j + "GGG" + "N"*j + "GGG" + "N"*j + "GGG"
#         seqs = []
#         for k in range(data_size-len(seq)):
#             s = "N"*(data_size-len(seq)-k) + seq + "N"*k
#             seqs.append(s)
#         p = make_prediction(model, seqs)
#         print(np.argmax(p, axis=0))
#         plt.plot(range(data_size-len(seq)), p)
#         plt.show()


def stretches_length_test(model, data_size, output):
    min_stretch = 1
    max_stretch = 10
    fig, ax = plt.subplots()
    seq_list = []
    for stretch_size in range(min_stretch, max_stretch + 1):
        seq = ("H" + "G"*stretch_size)*4 + "H"
        print(set_seq(seq, data_size, extra=1))
        seq_list.append(set_seq(seq, data_size, extra=1))
    stretch_predictions = make_prediction(model, seq_list)
    ax.plot(range(min_stretch, max_stretch + 1), stretch_predictions, 'o--')
    ax.set_xticks(range(min_stretch, max_stretch+1))
    plt.xlabel("Stretches length")
    plt.ylabel("rG4detector prediction")
    plt.xlim([min_stretch - 0.5, max_stretch + 0.5])
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.savefig(output + f"/Stretches influence", dpi=400)
    # TODO
    if PLOT:
        plt.show()
    # with open(output + f"Stretches influence.csv", "w") as f:
    #     f.write("stretch size, rG4detector prediction")
    #     for j, p in zip(range(min_stretch, max_stretch + 1), stretch_predictions):
    #         f.write(f"{j+1},{p}")




def main(model, output):
    data_size = get_input_size(model)
    loop_length_test(model, data_size, output)
    loop_length_test2(model, data_size, output)
    mutation_effect(model, data_size, output)
    seq = "CTGCTGCCGCTACTGCGGAGTAGCTGCTTCCCTTCCTCCTCTCCCGGCGGCGGCGGCGGCAGCGGCGGAGGAGGAGGAGGAGGGGACCCGGGCGCAGAGAGCCG" \
          "GCCGGCGGCGCAGTTGCAGCGCGGAG"
    mutation_map_test(model, data_size, output, seq)
    stretches_length_test(model, data_size, output)


if __name__ == "__main__":
    PLOT = True
    opts, args = getopt.getopt(sys.argv[1:], 'p')
    for op, val in opts:
        if op == "-p":
            PLOT = True

    MODEL = []
    for i in range(ENSEMBLE_SIZE):
        MODEL.append(load_model(MODEL_PATH + f"/model_{i}.h5"))
    main(MODEL, "../interpretation/")


