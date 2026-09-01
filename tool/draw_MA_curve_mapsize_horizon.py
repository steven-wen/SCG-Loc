import matplotlib.pyplot as plt
import random
from glob import glob

# 设置中文字体格式：微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


m_50 = [] 
m_20 = []
m_10 = []
m_5 = []
m_3 = []
m = [m_3,m_5,m_10,m_20,m_50]

y_label=[]
RDS_value = []
num_a=[3,5,10,20,50]
filenames = glob("output/result_files/mapsize-level/*.txt")
filenames.sort(key=lambda x:int(x.split("/")[-1].split("_")[0]))
for num, path in enumerate(filenames):
    y_label.append(path.split("/")[-1].split("_")[0])
    RDS_value.append(float(path.split("/")[-1].split(".txt")[0].split("=")[-1]))
    result = []
    with open(path, "r") as F:
        lines = F.readlines()
        for i,a in enumerate(num_a):
            line = lines[a-1]
            out = line.split(' ')[-1]
            out = out.split("\n")[0]
            m[i].append(float(out))

# y_label = ['700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600',"1700","1800"]
# y_label.reverse()
# 指定画布的尺寸与分辨率
plt.figure(figsize=(12, 9))


# 绘制横向条形图
bar1 = plt.bar(y_label, m_50, width=0.6, color="#45a776", label="MA@"+str(num_a[4]),edgecolor="k",linewidth=2)
bar2 = plt.bar(y_label, m_20, width=0.6, color="#3682be", label="MA@"+str(num_a[3]),edgecolor="k",linewidth=2)
bar3 = plt.bar(y_label, m_10, width=0.6, color="#b3974e", label="MA@"+str(num_a[2]),edgecolor="k",linewidth=2)
bar4 = plt.bar(y_label, m_5, width=0.6, color="#eed777", label="MA@"+str(num_a[1]),edgecolor="k",linewidth=2)
bar5 = plt.bar(y_label, m_3, width=0.6, color="#f05326", label="MA@"+str(num_a[0]),edgecolor="k",linewidth=2)

# 在横向条形图上添加数据
for (a, b) in zip(y_label, m_3):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(y_label, m_5):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(y_label, m_10):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(y_label, m_20):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(y_label, m_50):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
legend_bar = plt.legend(handles=[bar1,bar2,bar3,bar4,bar5], ncol=5, prop={'family': 'Times New Roman', 'size': 18})

# RDS_value = [0.675,0.717,0.742,0.754,0.787,0.793,0.804,0.787,0.782,0.777,0.740,0.749,]
rds, = plt.plot(y_label,RDS_value,color="red",linestyle="--", marker='*',label="RDS", markersize=14, linewidth=2)
legend_rds = plt.legend(handles=[rds],bbox_to_anchor = (0.85,0.85),prop={'family': 'Times New Roman', 'size': 18})
for ind, (a, b) in enumerate(zip(y_label, RDS_value)):
    if 7<=ind<=9:
        bias = -0.04
    else:
        bias = 0.01
    plt.text(a, b+bias, "{:.3f}".format(b), color='darkred', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')

ax = plt.gca().add_artist(legend_bar)

# plt.plot(y_label,m_3,color="black",linestyle="-")
# plt.plot(y_label, m_5, color="black", linestyle="-")
# plt.plot(y_label, m_10, color="black", linestyle="-")
# plt.plot(y_label, m_20, color="black", linestyle="-")
# plt.plot(y_label, m_50, color="black", linestyle="-")

# plt.title('3种品牌的销量对比', fontsize=16)  # 添加标题
plt.xticks(fontproperties='Times New Roman',fontsize=17)  # 添加y轴上的刻度
plt.yticks(fontproperties='Times New Roman', fontsize=17)  # 添加x轴上的标签
plt.ylim(0,1.07)
plt.ylabel(ylabel='Accurary', fontdict={'family': 'Times New Roman', 'size': 23})  # 添加x轴的标签
# plt.legend(loc="upper right", fontsize=16)  # 添加图例到右下角
plt.xlabel("Map Size",fontdict={'family' : 'Times New Roman', 'size': 23})

# 保存图像到当前文件夹下，图像名称为 image.png
plt.tight_layout()
plt.savefig('tool/visual/MA_curve/mapsize-level.jpg', dpi=600)
plt.savefig('tool/visual/MA_curve/mapsize-level.eps', dpi=600)
