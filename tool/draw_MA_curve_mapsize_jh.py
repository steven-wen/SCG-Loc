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
num_a=[3,5,10,20,50]
filenames = glob("output/result_files/mapsize-level/*.txt")
filenames.sort(key=lambda x:int(x.split("/")[-1].split("_")[0]))
for num, path in enumerate(filenames):
    y_label.append(path.split("/")[-1].split("_")[0])
    result = []
    with open(path, "r") as F:
        lines = F.readlines()
        for i,a in enumerate(num_a):
            line = lines[a-1]
            out = line.split(' ')[-1]
            out = out.split("\n")[0]
            m[i].append(int(float(out)*100))

# y_label = ['700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600',"1700","1800"]
# y_label.reverse()
# 指定画布的尺寸与分辨率
plt.figure(figsize=(9, 9))
_y = list(range(len(m_50)))

# 绘制横向条形图
plt.barh(_y, m_50, height=0.6, color="#ABDAB6", label="<"+str(num_a[4])+"m",edgecolor="k",linewidth=2)
plt.barh(_y, m_20, height=0.6, color="#7CCBC6", label="<"+str(num_a[3])+"m",edgecolor="k",linewidth=2)
plt.barh(_y, m_10, height=0.6, color="#4FB3D4", label="<"+str(num_a[2])+"m",edgecolor="k",linewidth=2)
plt.barh(_y, m_5, height=0.6, color="#FF9D9D", label="<"+str(num_a[1])+"m",edgecolor="k",linewidth=2)
plt.barh(_y, m_3, height=0.6, color="#FF7070", label="<"+str(num_a[0])+"m",edgecolor="k",linewidth=2)

# 在横向条形图上添加数据
# for (a, b, c) in zip(_y, positive, negative):
#        plt.text(b+6, a, str(b), color='g', fontsize=16)
#        plt.text(b+2, a, str(c), color='r', fontsize=16)

# plt.title('3种品牌的销量对比', fontsize=16)  # 添加标题
plt.yticks(_y, y_label, fontproperties='Times New Roman',fontsize=20)  # 添加y轴上的刻度
plt.xticks(fontproperties='Times New Roman', fontsize=20)  # 添加x轴上的标签
plt.xlim(0,100)
plt.xlabel(xlabel='Accurary(%)', fontdict={'family': 'Times New Roman', 'size': 20})  # 添加x轴的标签
# plt.legend(loc="upper right", fontsize=16)  # 添加图例到右下角
plt.ylabel("Mapsize",fontdict={'family' : 'Times New Roman', 'size': 20})
plt.legend(bbox_to_anchor=(0.93, 0.4), ncol=1,loc=3, borderaxespad=0, prop={'family': 'Times New Roman', 'size': 14})
# 保存图像到当前文件夹下，图像名称为 image.png
plt.tight_layout()
plt.savefig('tool/visual/MA_curve/mapsize-level.jpg', dpi=300)
