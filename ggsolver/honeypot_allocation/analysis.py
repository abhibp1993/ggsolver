import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main1():
    # with open("out/gw2_t0_f2/vod_fakes.pkl", "rb") as file:
    # with open("out/gw2_t2_f0/vod_traps.pkl", "rb") as file:
    with open("out/gw1_t1_f1/vod_traps.pkl", "rb") as file:
        data = pickle.load(file)

    vod_0 = np.ndarray((7, 7))
    vod_1 = np.ndarray((7, 7))

    for r in range(7):
        for c in range(7):
            if (r, c) in data[0]:
                vod_0[r, c] = data[0][(r, c)]
            else:
                vod_0[r, c] = 0.0

    for r in range(7):
        for c in range(7):
            if (r, c) in data[1]:
                vod_1[r, c] = data[1][(r, c)]
            else:
                vod_1[r, c] = 0.0

    fig, (ax0, ax1) = plt.subplots(1, 2)
    im0 = ax0.imshow(vod_0)
    im1 = ax1.imshow(vod_1)
    cbar0 = ax0.figure.colorbar(im0, ax=ax0, cmap="YlGn")
    cbar1 = ax1.figure.colorbar(im1, ax=ax1, cmap="YlGn")

    # Loop over data dimensions and create text annotations.
    for i in range(7):
        for j in range(7):
            text0 = ax0.text(j, i, round(vod_0[i, j], 3), ha="center", va="center", color="w")
            text1 = ax1.text(j, i, round(vod_1[i, j], 3), ha="center", va="center", color="w")

    ax0.set_title("VoD: Iteration 1")
    ax0.set_yticks(np.arange(7), labels=range(0, 7))
    # fig.tight_layout()
    # plt.savefig("out/gw2_t0_f2/vod_fakes_iter1.svg")

    ax1.set_title("VoD: Iteration 2")
    ax1.set_yticks(np.arange(7), labels=range(0, 7))
    # fig.tight_layout()
    plt.savefig("out/gw2_t0_f2/vod_fakes_iter2.svg")
    plt.show()


def main2():
    # with open("out/gw2_t2_f0/vod_traps.pkl", "rb") as file:
    # with open("out/gw2_t1_f1/vod_fakes.pkl", "rb") as file:
    # with open("out/gw2_t1_f1/vod_traps.pkl", "rb") as file:
    with open("out/gw2_t0_f2/vod_fakes.pkl", "rb") as file:
        data = pickle.load(file)

    vod_0 = np.ndarray((7, 7))
    vod_1 = np.ndarray((7, 7))

    for r in range(7):
        for c in range(7):
            if (r, c) in data[0]:
                vod_0[r, c] = data[0][(r, c)]
            else:
                vod_0[r, c] = 0.0

    for r in range(7):
        for c in range(7):
            if (r, c) in data[1]:
                vod_1[r, c] = data[1][(r, c)]
            else:
                vod_1[r, c] = 0.0

    # sns.heatmap(vod_0, annot=True, annot_kws={"size": 7})
    sns.heatmap(vod_1, annot=True, annot_kws={"size": 7})
    plt.savefig("6.svg")
    plt.show()


if __name__ == '__main__':
    # main1()
    main2()
