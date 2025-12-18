# SayHiDRL Training - CutLER Custom Dataset Training

This directory contains custom training scripts for CutLER, separated from the main CutLER library code.

## Structure

```
/workspace/
├── fbcutler/              # CutLER library (do not modify)
│   └── cutler/
│       └── ...
└── sayhidrl-training/     # Your custom training code (this directory)
    ├── register_custom_dataset.py
    ├── train_custom_dataset.py
    ├── cascade_mask_rcnn_R_50_FPN_custom.yaml
    └── ...
```

## Setup

The training scripts automatically reference the CutLER library located at `/workspace/fbcutler`. No additional setup is required.

## Usage

### 1. Register Your Dataset

```bash
cd /workspace/sayhidrl-training
python register_custom_dataset.py
```

This will register your datasets with Detectron2. Make sure to update the paths in `register_custom_dataset.py` to match your dataset location.

### 2. Train on Custom Dataset

```bash
python train_custom_dataset.py \
    --dataset-root /path/to/your/dataset \
    --train-images images/train \
    --train-json annotations/train.json \
    --val-images images/val \
    --val-json annotations/val.json \
    --num-gpus 8 \
    --output-dir output/my_training \
    --config-file cascade_mask_rcnn_R_50_FPN_custom.yaml
```

Or use the shell script:

```bash
bash train_custom_dataset.sh
```

## Files

- `register_custom_dataset.py` - Registers COCO-format datasets for training
- `train_custom_dataset.py` - Main training script with dataset registration
- `cascade_mask_rcnn_R_50_FPN_custom.yaml` - Training configuration file
- `train_custom_dataset.sh` - Shell script wrapper for training
- `TRAINING_CUSTOM_DATASET.md` - Detailed training documentation

## Notes

- The CutLER library is located at `/workspace/fbcutler` and is referenced via absolute paths
- All imports use `importlib` to avoid triggering CutLER's `__init__.py` which has import issues
- Dataset paths and configuration can be customized in the respective files
