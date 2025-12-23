import json
import copy
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm

# -------- Config --------
INPUT_JSON  = "../data/coco_kvasirseg/val/annotations/val.json"
OUTPUT_JSON = "../data/coco_kvasirseg/val/annotations/val.json"
MIN_AREA = 50   # bỏ instance quá nhỏ (noise)

# -------- Load --------
with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

new_annotations = []
new_ann_id = 1

# -------- Process --------
for ann in tqdm(coco["annotations"], desc="Splitting instances"):
    img_info = images[ann["image_id"]]
    h, w = img_info["height"], img_info["width"]

    # ---- Decode RLE ----
    rle = ann["segmentation"]
    mask = mask_utils.decode(rle)

    mask = (mask > 0).astype(np.uint8)

    # ---- Connected Components ----
    num_labels, labels = cv2.connectedComponents(mask)

    for label in range(1, num_labels):
        inst_mask = (labels == label).astype(np.uint8)

        if inst_mask.sum() < MIN_AREA:
            continue

        # ---- Encode back to RLE ----
        rle_inst = mask_utils.encode(
            np.asfortranarray(inst_mask)
        )
        rle_inst["counts"] = rle_inst["counts"].decode("ascii")

        area = int(mask_utils.area(rle_inst))
        bbox = mask_utils.toBbox(rle_inst).tolist()

        new_ann = {
            "id": new_ann_id,
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "segmentation": rle_inst,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }

        new_annotations.append(new_ann)
        new_ann_id += 1

# -------- Replace annotations --------
new_coco = copy.deepcopy(coco)
new_coco["annotations"] = new_annotations

# -------- Save --------
with open(OUTPUT_JSON, "w") as f:
    json.dump(new_coco, f, indent=2)

print(f"✅ Saved instance-level COCO to: {OUTPUT_JSON}")
print(f"Original anns: {len(coco['annotations'])}")
print(f"New anns: {len(new_annotations)}")
