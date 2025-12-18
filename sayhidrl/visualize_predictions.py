#!/usr/bin/env python
"""
Visualize model predictions and compare with ground truth annotations.

Usage:
    python visualize_predictions.py \
        --predictions output/kvasirseg_training/inference/coco_instances_results.json \
        --ground-truth ../coco_kvasirseg/eval/annotations/val.json \
        --images ../coco_kvasirseg/eval/images \
        --output-dir output/visualizations \
        --num-images 10
"""

import argparse
import json
import os
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

# Add cutler to path
cutler_root = "../fbcutler"
sys.path.insert(0, cutler_root)

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def load_predictions(pred_file):
    """Load predictions from COCO format JSON file."""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    return predictions


def decode_mask(segmentation, height=None, width=None):
    """Decode RLE or polygon mask to binary mask."""
    if isinstance(segmentation, dict):
        # COCO RLE format (dict with 'size' and 'counts')
        if 'size' in segmentation and 'counts' in segmentation:
            mask = maskUtils.decode(segmentation)
        else:
            # Try as polygon RLE (shouldn't happen with COCO format)
            if height is not None and width is not None:
                rle = maskUtils.frPyObjects([segmentation], height, width)
                mask = maskUtils.decode(rle)
            else:
                raise ValueError("Height and width required for polygon RLE")
    elif isinstance(segmentation, list):
        # Polygon format (list of polygons)
        if height is None or width is None:
            raise ValueError("Height and width required for polygon format")
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            # List of polygons
            rle = maskUtils.frPyObjects(segmentation, height, width)
            mask = maskUtils.decode(rle)
        else:
            # Single polygon as list
            rle = maskUtils.frPyObjects([segmentation], height, width)
            mask = maskUtils.decode(rle)
    else:
        raise ValueError(f"Unknown segmentation format: {type(segmentation)}")
    
    # Handle multi-channel masks
    if len(mask.shape) == 3:
        mask = mask.sum(axis=2) > 0
    elif len(mask.shape) == 2:
        mask = mask > 0
    
    return mask.astype(np.uint8) * 255


def vis_mask(img, mask, color=[200, 0, 0], alpha=0.4):
    """
    Overlay mask on image.
    img: (H, W, 3) uint8 RGB
    mask: (H, W) uint8, {0, 255} or {0,1}
    color: (3,) RGB
    alpha: transparency (0 = transparent, 1 = opaque)
    """
    out = img.copy().astype(np.float32)
    
    if mask.max() > 1:
        mask_bool = mask > 0
    else:
        mask_bool = mask == 1
    
    color = np.array(color, dtype=np.float32)
    
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color
    
    return out.astype(np.uint8)


def visualize_comparison(image_path, gt_annotations, pred_annotations, output_path, score_threshold=0.5):
    """
    Visualize predictions and ground truth side by side.
    
    Args:
        image_path: Path to the image file
        gt_annotations: List of ground truth annotations for this image
        pred_annotations: List of prediction annotations for this image
        output_path: Path to save the visualization
        score_threshold: Minimum confidence score to show predictions
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Create figure with 3 subplots: original, ground truth, predictions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth overlay
    img_gt = img_rgb.copy()
    gt_count = 0
    for ann in gt_annotations:
        if 'segmentation' in ann and ann['segmentation']:
            try:
                seg = ann['segmentation']
                # Decode mask (COCO RLE format includes size info)
                mask = decode_mask(seg)
                
                # Resize mask if dimensions don't match image
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                img_gt = vis_mask(img_gt, mask, color=[0, 255, 0], alpha=0.5)  # Green for GT
                gt_count += 1
            except Exception as e:
                print(f"Warning: Could not decode GT mask for {os.path.basename(image_path)}: {e}")
                continue
        
        # Draw bounding box
        if 'bbox' in ann and ann['bbox']:
            bbox = ann['bbox']  # [x, y, width, height]
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                           linewidth=2, edgecolor='green', facecolor='none')
            axes[1].add_patch(rect)
    
    axes[1].imshow(img_gt)
    axes[1].set_title(f'Ground Truth ({gt_count} objects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Predictions overlay
    img_pred = img_rgb.copy()
    pred_count = 0
    for ann in pred_annotations:
        score = ann.get('score', 1.0)
        if score < score_threshold:
            continue
        
        if 'segmentation' in ann and ann['segmentation']:
            try:
                seg = ann['segmentation']
                # Decode mask (COCO RLE format includes size info)
                mask = decode_mask(seg)
                
                # Resize mask if dimensions don't match image
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                img_pred = vis_mask(img_pred, mask, color=[255, 0, 0], alpha=0.5)  # Red for predictions
                pred_count += 1
            except Exception as e:
                print(f"Warning: Could not decode prediction mask: {e}")
                continue
        
        # Draw bounding box with score
        if 'bbox' in ann and ann['bbox']:
            bbox = ann['bbox']  # [x, y, width, height]
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
            # Add score text
            axes[2].text(bbox[0], bbox[1] - 5, f'{score:.2f}', 
                        color='red', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[2].imshow(img_pred)
    axes[2].set_title(f'Predictions ({pred_count} objects, score>{score_threshold})', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def main(args):
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Load ground truth COCO format
    print(f"Loading ground truth from {args.ground_truth}...")
    coco_gt = COCO(args.ground_truth)
    
    # Get image IDs
    img_ids = coco_gt.getImgIds()
    if args.num_images > 0:
        img_ids = img_ids[:args.num_images]
    
    print(f"Processing {len(img_ids)} images...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Group predictions by image_id
    pred_by_image = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)
    
    # Process each image
    for img_id in img_ids:
        # Get image info
        img_info = coco_gt.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(args.images, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Get ground truth annotations
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        
        # Get predictions for this image
        pred_anns = pred_by_image.get(img_id, [])
        
        # Create visualization
        output_filename = os.path.splitext(img_filename)[0] + '_comparison.png'
        output_path = os.path.join(args.output_dir, output_filename)
        
        visualize_comparison(img_path, gt_anns, pred_anns, output_path, args.score_threshold)
    
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions and compare with ground truth")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file (COCO format)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth annotations JSON file (COCO format)",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations",
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to visualize (0 for all)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to show predictions",
    )
    
    args = parser.parse_args()
    main(args)

