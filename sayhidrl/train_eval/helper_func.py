import os, shutil
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import numpy as np
from pycocotools import mask as maskUtils

def extract_first_images(dataset_name, test_number):
    """
    Extract first test_number images from train and val datasets (in coco format)and save them in a new path
    """
    # Get the full dataset and limit it
    full_dataset = DatasetCatalog.get(dataset_name)
    limited_dataset = full_dataset[:min(test_number, len(full_dataset))]
    
    # Register a limited version
    limited_dataset_name = dataset_name + "_limited"
    DatasetCatalog.register(limited_dataset_name, lambda: limited_dataset)
    
    # Copy metadata from original dataset to limited dataset
    original_metadata = MetadataCatalog.get(dataset_name)
    limited_metadata = MetadataCatalog.get(limited_dataset_name)
    
    # Copy all metadata attributes dynamically (skip 'name' as it's read-only)
    skip_attrs = {'name'}  # Attributes that should not be copied
    for attr_name in dir(original_metadata):
        if (not attr_name.startswith('_') and 
            attr_name not in skip_attrs and
            not callable(getattr(original_metadata, attr_name, None))):
            try:
                attr_value = getattr(original_metadata, attr_name)
                setattr(limited_metadata, attr_name, attr_value)
            except (AttributeError, TypeError, AssertionError):
                pass  # Skip attributes that can't be copied
    print(f"Limited test dataset to {len(limited_dataset)} images (from {len(full_dataset)} total)")
    return limited_dataset_name


## ============== helper function to read images in coco format ======================
def load_image(image_path):
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def decode_rle_mask(rle_dict):
    """Decode RLE mask in coco format to binary mask."""
    if isinstance(rle_dict, dict):
        return maskUtils.decode(rle_dict)
    elif isinstance(rle_dict, list):
        # Polygon format - convert to RLE first 
        h, w = rle_dict[0]['size'] if isinstance(rle_dict[0], dict) else (1000, 1000)
        return maskUtils.decode(maskUtils.frPyObjects(rle_dict, h, w))
    return None

def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    """Draw bounding box on image. bbox is [x, y, width, height] in COCO format."""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def draw_mask(img, mask, color=(255, 0, 0), alpha=0.4):
    """Overlay mask on image."""
    if mask is None:
        return img
    
    if mask.max() > 1:
        mask = mask > 0
    else:
        mask = mask == 1
    
    color = np.array(color, dtype=np.float32)
    out = img.copy().astype(np.float32)
    
    mask_area = mask > 0
    out[mask_area] = (1 - alpha) * out[mask_area] + alpha * color
    
    return out.astype(np.uint8)


def draw_annotations_on_image(img, annotations, color=(0, 255, 0), show_scores=False):
    """Draw annotations (bboxes and masks) on image."""
    result_img = img.copy()
    
    for ann in annotations:
        # Draw bbox
        if 'bbox' in ann:
            bbox = ann['bbox']  # [x, y, w, h]
            result_img = draw_bbox(result_img, bbox, color=color, thickness=2)
            
            # Add score text if available
            if show_scores and 'score' in ann:
                x, y, w, h = bbox
                x, y = int(x), int(y)
                score = ann['score']
                cv2.putText(result_img, f'{score:.2f}', (x, max(y-5, 0)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw mask if available
        if 'segmentation' in ann:
            mask = decode_rle_mask(ann['segmentation'])
            if mask is not None:
                result_img = draw_mask(result_img, mask, color=color, alpha=0.4)
    
    return result_img