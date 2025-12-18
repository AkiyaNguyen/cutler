#!/bin/bash
# Example training script for custom dataset

# Make sure to register your dataset first by importing register_custom_dataset.py
# You can do this by adding: import register_custom_dataset
# Or by running: python -c "import register_custom_dataset"

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLER_ROOT="/workspace/fbcutler"

# Set number of GPUs
NUM_GPUS=8

# Set paths - use local config file or fallback to cutler's demo config
CONFIG_FILE="${SCRIPT_DIR}/cascade_mask_rcnn_R_50_FPN_custom.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
    CONFIG_FILE="${CUTLER_ROOT}/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml"
fi

WEIGHTS="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_r2.pth"
OUTPUT_DIR="output/custom_dataset_training"

# Run training using cutler's train_net.py
python ${CUTLER_ROOT}/cutler/train_net.py \
  --num-gpus ${NUM_GPUS} \
  --config-file ${CONFIG_FILE} \
  MODEL.WEIGHTS ${WEIGHTS} \
  DATASETS.TRAIN "('your_dataset_train',)" \
  DATASETS.TEST "('your_dataset_val',)" \
  OUTPUT_DIR ${OUTPUT_DIR}

# Alternative: You can also override config values via command line:
# python train_net.py \
#   --num-gpus 8 \
#   --config-file ${CONFIG_FILE} \
#   MODEL.WEIGHTS ${WEIGHTS} \
#   DATASETS.TRAIN "('your_dataset_train',)" \
#   DATASETS.TEST "('your_dataset_val',)" \
#   SOLVER.BASE_LR 0.002 \
#   SOLVER.MAX_ITER 50000 \
#   OUTPUT_DIR ${OUTPUT_DIR}

