# SCG-LOC

SCG-LOC 是一个与论文对应的 UAV image–satellite image 跨视角地理定位项目。代码以一张 UAV image 和一张 satellite image 为输入，通过骨干网络提取多尺度特征，经过颈部、结构匹配/引导、定位头与后处理模块，输出位置响应图及可选的亚像素偏移，并计算 RDS、SDM 和不同距离阈值下的定位准确率。

## 仓库内容与当前状态

本仓库按以下原则组织：

- 提交论文对应的项目源代码、运行配置和说明文档。
- 提交由程序直接生成、能够与论文表格或结论对应的最终指标文件。
- 不提交 Python/Conda 环境目录、数据集、预训练权重、训练检查点、日志、中间预测、缓存、token、密钥或本机私有配置。

当前版本已经包含模型、数据读取、损失函数、训练、测试和汇总评估等主体代码。本机阶段只完成仓库的基础整理，不安装服务器环境，也不执行训练或测试。当前设备没有论文最终结果文件，因此仓库中暂不包含结果数值；后续在服务器上由程序重新生成后再提交，不能手工填写。

## 项目结构

```text
SCG-LOC/
├── SCG-LOC.py                # 当前模型、数据、训练和测试配置
├── train.py                  # 训练入口
├── test_meter.py             # 测试及 RDS/SDM/MA 指标生成
├── evaluate_all.py           # 按总体、地图尺寸、范围和高度汇总结果
├── heatmap.py                # 响应热图可视化工具
├── train_test_local.sh       # Linux 训练/测试串行脚本
├── datasets/                 # 数据读取和增强
├── models/
│   ├── Backbone/             # 特征提取骨干网络
│   ├── Neck/                 # 多尺度特征融合及结构引导模块
│   ├── Head/                 # 匹配与定位预测头
│   └── PostProcess/          # 响应图上采样等后处理
├── losses/                   # 分类、回归和定位损失
├── optimizers/               # 优化器和学习率策略
└── output/
    ├── metric_RDS_SDM/       # 可提交：程序生成的总体指标
    ├── result_files/         # 可提交：程序生成的论文结果汇总
    └── pred_GPS/             # 不提交：逐样本中间预测
```


## 安装

`requirements.txt` 只记录从当前源码中能够确认的基础 Python 依赖，不代表完整、已验证的服务器环境。后续在服务器上使用独立的 Python 环境；环境目录本身不要提交到仓库。

1. 创建并激活本地环境，例如：

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. 根据服务器 CUDA 版本安装相互匹配的 PyTorch 与 torchvision。

3. 安装基础 Python 依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. 本项目使用 MMCV 1.x 风格的 `Config`/`runner` 接口，并有 MMDetection 2.x 风格的导入。在服务器上按照实际 PyTorch 和 CUDA 版本安装兼容的 `mmcv-full<2` 与 `mmdet<3`。

本机不验证上述运行环境。服务器执行阶段再确定并记录实际使用的 Python、PyTorch、CUDA、MMCV 和 MMDetection 版本。

## 数据准备

数据集不纳入 Git。后续在服务器准备数据，并修改 `SCG-LOC.py` 中以下路径：

- `data_config.train_dir`
- `data_config.val_dir`
- `data_config.test_dir`
- `data_config.test_mode`
- `model.backbone.pretrain_path`（如使用预训练权重）

代码期望每个样本目录至少具有以下结构：

```text
sample/
├── UAV/0.JPG
├── Satellite/<satellite image files>
├── labels.json
└── GPS_info.json              # 测试和米级评估需要
```

`labels.json` 保存 satellite image 文件名对应的目标像素坐标；`GPS_info.json` 保存 UAV image 与 satellite image 的地理坐标信息。实际字段要求可查看 `datasets/SiamUAV.py`。

## 使用方式

以下命令应在项目根目录执行。

### 训练

先在 `SCG-LOC.py` 中设置数据与预训练权重路径，然后执行：

```bash
python train.py --config SCG-LOC.py --name scg_loc
```

训练检查点、TensorBoard 记录和日志写入 `checkpoints/`，这些文件属于环境相关或中间产物，默认不提交。

### 测试

将 `SCG-LOC.py` 中 `test_config.checkpoint` 设置为程序生成的检查点路径，然后执行：

```bash
python test_meter.py --config SCG-LOC.py
```

该程序会生成：

- `output/metric_RDS_SDM/`：RDS、SDM、MA 与耗时汇总；
- `output/pred_GPS/`：逐样本 GPS 预测，属于后续汇总所需的中间结果。

### 生成论文对应的汇总结果

```bash
python evaluate_all.py \
  --info_file output/pred_GPS/GPS_pred_gt_filterR-33.json \
  --out_file output/result_files/
```

`output/result_files/` 中的总体和分组指标，以及 `output/metric_RDS_SDM/` 中与论文对应的最终指标，可以在核对实验配置、数据版本和检查点后提交。`output/pred_GPS/`、检查点和日志继续保持忽略。

## 提交前检查

```bash
git status --short
git check-ignore -v output/result_files/total-level.txt
git check-ignore -v output/pred_GPS/predictions.json
```

预期行为：最终指标文件不被忽略，逐样本预测被忽略。提交前还应确认配置中不存在本机绝对路径、账号、token、密钥或未获许可的数据。
