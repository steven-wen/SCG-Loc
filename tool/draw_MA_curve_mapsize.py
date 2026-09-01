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
    index = [0,1,2,3,5,7,10,20,30,40,50,60,70,80,90,99]
    for line in np.array(context)[index]:
        m, p = line.split("\n")[0].split(" ")
        m = float(m)
        p = float(p)
        meter.append(m)
        prob.append(p)
    return meter, prob


root_dir = "./result_files/circle-level"

if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 4))
    # plt.grid()

    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'maroon', 'peru', 'teal']

    plt.yticks(fontproperties='Times New Roman', size=8)
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.xlim(0,100)
    plt.ylim(0,1)

    filenames = glob.glob(os.path.join(root_dir, "*.txt"))
    filenames.sort()
    for ind, filename in enumerate(filenames):
        meter, prob = read_from_txt(filename)
        cubic_interploation_model=interp1d(meter,prob,kind="cubic")
        xs=np.linspace(0,100,500)
        ys=cubic_interploation_model(xs)
        label = filename.split("/")[-1].split(".txt")[0]
        plt.plot(xs, ys, c=color_list[ind], label=label)

    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 5})
    plt.ylabel("AP(%)",fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.xlabel("numbers of sampling k",fontdict={'family' : 'Times New Roman', 'size': 16})
    # plt.tight_layout()
    fig.savefig('test.png', dpi=300)
    # plt.show()


    # plt.plot(x_data,y_data,color="pink",linestyle="--")
    # plt.plot(x_data, y2_data+200, color="skyblue", linestyle="-.")
    
    # # 柱状图
    # plt.bar(x_data,y_data,lw=0.5,fc="r",width=0.3,label="Phone",alpha=0.5)
    # plt.bar(x_data,y2_data, lw=0.5, fc="b", width=0.3, label="Android",alpha=0.5,bottom=y_data)
    
    # for i,j in zip(x_data,y_data):
    
    # plt.text(i,j+0.05,"%d"%j,ha="center",va="bottom")
    
    # for i2,j2 in zip(x_data,y2_data):
    
    # plt.text(i2,j2+180,"%d"%j2,ha="center",va="bottom")

