#!/usr/bin/env python
"""
Convert Kvasir-SEG to COCO instance format with train/eval split.
"""

import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import shutil
import random

# ---------------- Config ----------------
DATA_ROOT = "/workspace/kvasirseg/Kvasir-SEG/Kvasir-SEG"
IMG_SRC_DIR = os.path.join(DATA_ROOT, "images")
MASK_SRC_DIR = os.path.join(DATA_ROOT, "masks")

OUTPUT_ROOT = "/workspace/coco_kvasirseg"
TRAIN_RATIO = 0.88
SEED = 42
random.seed(SEED)

# ---------------- Helper Functions ----------------
def mask_to_coco_annotation(mask, image_id, annotation_id):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    area = int(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }

def create_coco_json(image_files, src_img_dir, src_mask_dir, dst_img_dir, output_json):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp"}]
    }
    ann_id = 1
    for img_id, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        # Copy image
        shutil.copyfile(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

        # Add image info
        im = cv2.imread(os.path.join(src_img_dir, img_file))
        height, width = im.shape[:2]
        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_file,
            "height": height,
            "width": width
        })

        # Load mask
        mask_file = img_file
        mask_path = os.path.join(src_mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bin = (mask > 127).astype(np.uint8)

        # Create annotation
        ann = mask_to_coco_annotation(mask_bin, img_id, ann_id)
        coco_dict["annotations"].append(ann)
        ann_id += 1

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"Saved COCO JSON to {output_json}")

# ---------------- Main ----------------
if __name__ == "__main__":
    all_images = sorted(os.listdir(IMG_SRC_DIR))
    n_total = len(all_images)
    n_train = int(TRAIN_RATIO * n_total)

    random.shuffle(all_images)
    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:]

    # Directories
    train_img_dir = os.path.join(OUTPUT_ROOT, "train/images")
    val_img_dir   = os.path.join(OUTPUT_ROOT, "eval/images")

    train_json = os.path.join(OUTPUT_ROOT, "train/annotations/train.json")
    val_json   = os.path.join(OUTPUT_ROOT, "eval/annotations/val.json")

    # Create COCO JSON
    create_coco_json(train_imgs, IMG_SRC_DIR, MASK_SRC_DIR, train_img_dir, train_json)
    create_coco_json(val_imgs, IMG_SRC_DIR, MASK_SRC_DIR, val_img_dir, val_json)
