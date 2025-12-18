#!/bin/bash
# Script to visualize predictions and compare with ground truth

python visualize_predictions.py \
    --predictions output/kvasirseg_training/inference/coco_instances_results.json \
    --ground-truth ../coco_kvasirseg/eval/annotations/val.json \
    --images ../coco_kvasirseg/eval/images \
    --output-dir output/visualizations \
    --num-images 10 \
    --score-threshold 0.5

