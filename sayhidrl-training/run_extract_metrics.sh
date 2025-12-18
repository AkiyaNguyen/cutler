#!/bin/bash
# Script to extract and display evaluation metrics

python extract_metrics.py \
    --predictions output/kvasirseg_training/inference/coco_instances_results.json \
    --ground-truth /workspace/coco_kvasirseg/eval/annotations/val.json \
    --output-file output/metrics.json

