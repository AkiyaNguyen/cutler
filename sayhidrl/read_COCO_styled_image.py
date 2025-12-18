import json
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

def vis_mask(img, mask, color=[200, 0, 0], alpha=0.4):
    """
    img: (H, W, 3) uint8
    mask: (H, W) uint8, {0, 255} or {0,1}
    color: (3,) RGB
    alpha: transparency (0 = transparent, 1 = opaque)
    """
    out = img.copy()

    if mask.max() > 1:
        mask = mask > 0
    else:
        mask = mask == 1

    color = np.array(color, dtype=np.float32)

    out[mask] = (
        (1 - alpha) * out[mask] +
        alpha * color
    )

    return out.astype(np.uint8)

def read_COCO_styled_image(json_path, img_folder_path, output_folder_path, num_images=5):
    """
    json_path: path to the json file
    img_folder_path: path to the image folder
    output_folder_path: path to the output folder
    num_images: number of images to read
    """

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    with open(json_path, "r") as f:
        coco = json.load(f)

    coco["images"] = sorted(coco["images"], key=lambda x: x["id"])
    coco["annotations"] = sorted(coco["annotations"], key=lambda x: x["id"])
    
    
    for i in range(min(num_images, len(coco["images"]))):
        img_path = img_folder_path + coco["images"][i]["file_name"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = coco["annotations"][i]["segmentation"]
        mask = maskUtils.decode(mask)   
        overlay = vis_mask(img, mask, color=[200, 0, 0])

        name = '_'.join(coco["images"][i]["file_name"].split('/'))
        cv2.imwrite(output_folder_path + name, overlay)
        print("Saved overlay image to ", output_folder_path + name)

if __name__ == "__main__":
    json_path = "/workspace/polybOutput/imagenet_train_fixsize480_tau0.15_N3.json"
    img_folder_path = "/workspace/polybDataset/"
    output_folder_path = "/workspace/polybOutput/maskcut/"
    num_images = 5
    read_COCO_styled_image(json_path, img_folder_path, output_folder_path, num_images)