# SCG-Loc

[English](README.md) | [简体中文](README_zh-CN.md)

SCG-Loc is a cross-view geo-localization framework for matching UAV imagery with satellite imagery. Given a UAV query and a satellite image, the model predicts the query location on the satellite map and evaluates it with RDS, SDM, and meter-level accuracy metrics.

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

You need to prepare the UL14 dataset and pre-trained weights in advance.

## Usage

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config SCG-LOC.py \
  --gpu_ids 0 \
  --name scg_loc
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test_meter.py --config SCG-LOC.py
```

### Result aggregation

```bash
python evaluate_all.py \
  --info_file output/pred_GPS/GPS_pred_gt_filterR-33.json \
  --out_file output/result_files/
```

### Train and evaluate

```bash
chmod +x train_test_local.sh
./train_test_local.sh SCG-LOC.py
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
