# SCG-Loc: A Structure-Calibrated Geo-Localization Network for Pixel-Level UAV Localization

[English](README.md)

SCG-Loc is a cross-view geo-localization framework for matching UAV imagery with satellite imagery. Given a UAV query and a satellite image, the model predicts the query location on the satellite map and evaluates it with RDS, SDM, and meter-level accuracy metrics.

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training and Testing](#training-and-testing)
  - [Evaluation](#evaluation)
  - [Result Aggregation](#result-aggregation)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [License](#license)


## Highlights

- **DPFPN — Detail-Preserving Feature Pyramid Network.** Builds detail-aware multi-scale representations with three components:
  - **Receptive Field Aggregation** for local, dilated, and global context.
  - **Fusion Gate** for adaptive lateral and top-down feature fusion.
  - **Detail Residual Extractor** for restoring fine-grained spatial cues.
- **DSPM — Dynamic Structural Prototype Matching.** Builds a structural prototype from UAV features, performs local structural matching on satellite features, and injects the matched context into the high-resolution feature map.
- End-to-end training and evaluation with response-map decoding and optional location-offset refinement.

## Architecture

```text
UAV image       ── Shared MixFormer backbone ── DPFPN ──┐
                                                        ├── DSPM ── Matching head ── Response map
Satellite image ── Shared MixFormer backbone ── DPFPN ──┘
```

## Installation

The reference environment is named `scg-loc` and uses Python 3.9, PyTorch 2.0.0, CUDA 11.8, MMCV 1.7.2, and MMDetection 2.28.2.

```bash
conda create -n scg-loc python=3.9.23 -y
conda activate scg-loc

python -m pip install \
  torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

python -m pip install openmim==0.3.9
mim install mmcv-full==1.7.2
python -m pip install -r requirements.txt
```

## Data Preparation

This project uses the **UL14** dataset, following the same dataset setup as [DRL](https://github.com/Dmmm1997/DRL).

- **UL14 dataset:** [Download from ModelScope](https://modelscope.cn/datasets/dmmm997/UL14/files)
- **CvT-13 pre-trained weights:** [Download pre-trained weights](https://seunic-my.sharepoint.cn/:u:/g/personal/230238525_seu_edu_cn/EUnD6wdRZ3hKhefxpqBsdpIBqzxzFGN7mm2rL548QZXtZw?e=YvwDFJ)

After downloading the dataset and pre-trained weights, update the corresponding paths in the configuration file:

- `pretrain_path`
- `train_dir`
- `val_dir`
- `test_dir`
## Usage

### Training and Testing

Execute the following command to run the training and testing pipeline:

```bash
bash train_test_local.sh
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test_meter.py --config SCG-LOC.py
```

### Result Aggregation

```bash
python evaluate_all.py \
  --info_file output/pred_GPS/GPS_pred_gt_filterR-33.json \
  --out_file output/result_files/
```
## Project Structure

```text
SCG-Loc/
├── SCG-LOC.py              
├── train.py                
├── test_meter.py           
├── evaluate_all.py         
├── datasets/               
├── models/
│   ├── Backbone/           
│   ├── Neck/
│   │   ├── dpfpn.py       
│   │   └── dspm.py        
│   ├── Head/               
│   ├── PostProcess/        
│   └── taskflow.py         
├── losses/                 
├── optimizers/             
├── tool/                   
└── output/                 
```

## Outputs

Evaluation writes metrics and predictions under `output/`:

```text
output/
├── metric_RDS_SDM/
├── pred_GPS/
└── result_files/
```

## License

This repository does not currently include a license file. Add an appropriate open-source license before redistribution.
