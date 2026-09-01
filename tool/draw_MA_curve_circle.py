import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import glob
import os
import numpy as np
from scipy.interpolate import interp1d


def read_from_txt(filename):
    with open(filename, "r") as f:
        context = f.readlines()
    meter, prob = [], []
    index = list(range(30))
    for line in np.array(context)[index]:
        m, p = line.split("\n")[0].split(" ")
        m = float(m)
        p = float(p)
        meter.append(m)
        prob.append(p)
    return meter, prob


root_dir = "./output/result_files/circle-level"

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 6))
    plt.grid()

    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'maroon', 'peru', 'teal']

    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlim(0,30)
    plt.ylim(0,1)

    filenames = glob.glob(os.path.join(root_dir, "*.txt"))
    filenames.sort()
    for ind, filename in enumerate(filenames):
        meter, prob = read_from_txt(filename)
        basename = os.path.basename(filename)
        label = basename.split("_")[0]
        plt.plot(meter, prob, c=color_list[ind], label=label)

    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 16})
    plt.ylabel("MA@K",fontdict={'family' : 'Times New Roman', 'size': 17})
    plt.xlabel("Spatial Distance Threshold K (m)",fontdict={'family' : 'Times New Roman', 'size': 17})
    fig.savefig('./tool/visual/MA_curve/circle-level.jpg', dpi=600, format="jpg")
    fig.savefig('./tool/visual/MA_curve/circle-level.eps', dpi=600, format="eps")

