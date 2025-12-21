#!/usr/bin/env python
"""
Example script to register a custom dataset for CutLER training.

Usage:
    python register_custom_dataset.py

Then use the registered dataset name in your config file or command line.
"""

import os
import sys
import importlib.util
from detectron2.data import MetadataCatalog

# Add the cutler package to path
# CutLER library is located at ../fbcutler
cutler_root = "../fbcutler"
sys.path.insert(0, cutler_root)

# Import register_coco_instances directly without triggering cutler/__init__.py
coco_module_path = os.path.join(cutler_root, "cutler", "data", "datasets", "coco.py")
spec = importlib.util.spec_from_file_location("coco_module", coco_module_path)
coco_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coco_module)
register_coco_instances = coco_module.register_coco_instances

# Set your dataset root directory
# You can also use environment variable: DETECTRON2_DATASETS
DATASET_ROOT = os.path.expanduser("../coco_kvasirseg")

# Example: Register a custom dataset
# Replace these with your actual paths
CUSTOM_DATASET_NAME = "kvasirseg_train"
CUSTOM_DATASET_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
CUSTOM_DATASET_JSON = os.path.join(DATASET_ROOT, "train", "annotations", "train.json")

# Register the dataset
register_coco_instances(
    CUSTOM_DATASET_NAME,
    {},  # Empty metadata dict - will be auto-populated from COCO JSON
    CUSTOM_DATASET_JSON,
    CUSTOM_DATASET_IMAGES,
)

# Optionally register validation set
CUSTOM_DATASET_VAL_NAME = "kvasirseg_val"
CUSTOM_DATASET_VAL_IMAGES = os.path.join(DATASET_ROOT, "val", "images")
CUSTOM_DATASET_VAL_JSON = os.path.join(DATASET_ROOT, "val", "annotations", "val.json")

register_coco_instances(
    CUSTOM_DATASET_VAL_NAME,
    {},
    CUSTOM_DATASET_VAL_JSON,
    CUSTOM_DATASET_VAL_IMAGES,
)

print(f"Registered dataset: {CUSTOM_DATASET_NAME}")
print(f"Registered dataset: {CUSTOM_DATASET_VAL_NAME}")

# You can also add custom metadata if needed
# metadata = MetadataCatalog.get(CUSTOM_DATASET_NAME)
# metadata.thing_classes = ["class1", "class2", ...]  # Your class names
# metadata.evaluator_type = "coco"  # Use COCO evaluator

