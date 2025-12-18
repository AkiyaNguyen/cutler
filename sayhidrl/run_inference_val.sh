#!/bin/bash
# Script to run inference/evaluation on validation set

# Set your checkpoint path here (update this to your trained model)
CHECKPOINT_PATH="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_r2.pth"

# Run inference
# Set NUM_TEST_IMAGES to limit the number of images (e.g., 10 for quick testing)
# Leave empty or set to 0 to process all images
NUM_TEST_IMAGES=10

python train_custom_dataset.py \
    --eval-only \
    --cpu \
    --dataset-root /workspace/coco_kvasirseg \
    --val-images eval/images \
    --val-json eval/annotations/val.json \
    --config-file cascade_mask_rcnn_R_50_FPN_custom.yaml \
    --checkpoint "${CHECKPOINT_PATH}" \
    --num-test-images ${NUM_TEST_IMAGES} \
    --num-gpus 0

