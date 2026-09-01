import matplotlib.pyplot as plt
import random
from glob import glob
import numpy as np

fig = plt.figure(figsize=(8, 7))



x_label = ["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0",">1.0"]


ax1 = fig.subplots()

ax1.tick_params(axis='x', labelrotation=10)

plt.xlabel("Range",fontdict={'family' : 'Times New Roman', 'size': 22})
plt.xticks(fontproperties='Times New Roman',fontsize=18)

Num_value = [944,2133,4900,6655,9012,4328]

ax1.bar(x_label, Num_value, color="#4e954c", label="Sample Number")
ax1.set_ylabel('Sample Number', fontdict={'family': 'Times New Roman', 'size': 22})
ax1.set_ylim(0,10200)
ax1.set_yticklabels(list(range(0,10200,2000)),fontproperties='Times New Roman', fontsize=16)  # 添加x轴上的标签
for ind, (a, b) in enumerate(zip(x_label, Num_value)):
    bias = + 0.001
    plt.text(a, b+bias, "{:d}".format(b), color='black', fontsize=18, ha="center",va="bottom", fontproperties='Times New Roman')


ax2 = ax1.twinx()

RDS_value = [0.750, 0.760, 0.766, 0.786, 0.766, 0.694]
ax2.plot(x_label,RDS_value,color="red",linestyle="--", marker='*',label="RDS", markersize=16, linewidth=3)
# legend_rds = plt.legend(prop={'family': 'Times New Roman', 'size': 20})
for ind, (a, b) in enumerate(zip(x_label, RDS_value)):
    bias = + 0.003
    plt.text(a, b+bias, "{:.3f}".format(b), color='darkred', fontsize=18, ha="center",va="bottom", fontproperties='Times New Roman')

ax2.set_ylabel('RDS', fontdict={'family': 'Times New Roman', 'size': 22})  # 添加x轴的标签
ax2.set_ylim(0.67,0.82)
# ax1.set_xticks(fontproperties='Times New Roman',fontsize=22)  # 添加y轴上的刻度


# ax2.set_xticks(fontproperties='Times New Roman',fontsize=20)  # 添加y轴上的刻度
ax2.set_yticklabels([0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84], fontproperties='Times New Roman', fontsize=18)  # 添加x轴上的标签

fig.legend(bbox_to_anchor = (0.49,0.965), prop={'family': 'Times New Roman', 'size': 18})  # 添加图例到右下角

# 保存图像到当前文件夹下，图像名称为 image.png
plt.tight_layout()
plt.savefig('tool/visual/MA_curve/RDS_circle.jpg', dpi=600)
plt.savefig('tool/visual/MA_curve/RDS_circle.eps', dpi=600)
