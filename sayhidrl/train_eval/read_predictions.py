"""
example usage:
python read_predictions.py --coco_instance_result_file ...
                            --num_images ...
                            --image_path ...
                            --annotation_file ...
                            --output_dir ...
Running this script to analyze the result of model predictions displayed in coco format.
This would plot pr_curve, plot the predictions and the ground truth and compute the ap scores.
"""
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

from helper_func import load_image, draw_annotations_on_image



def plot_gt_and_predictions(img, gt, predList, image_path, output_dir, threshold):
    """
    Plot ground truth and predictions for a single image.
    
    Args:
        img: dict with id, file_name, height, width
        gt: list of ground truth annotations
        predList: list of predictions
        image_path: base path to images directory
        output_dir: directory to save the visualization
        threshold: threshold for predictions
    Creates 3 panels:
    - Original image
    - Ground truth annotations
    - Predictions
    """
    # Load the image
    full_image_path = os.path.join(image_path, img['file_name'])
    original_img = load_image(full_image_path)
    
    # Create visualization with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(0.02, 0.98, f"ID: {img.get('id', 'N/A')}\n{img.get('file_name', '')}", 
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Ground truth (green)
    gt_img = draw_annotations_on_image(original_img, gt, color=(0, 255, 0))
    axes[1].imshow(gt_img)
    axes[1].set_title(f'Ground Truth ({len(gt)} objects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Predictions (red), only show predictions with score >= threshold
    max_score = np.max([pred['score'] for pred in predList])
    if threshold > max_score:
        threshold = max_score - 1e-6
    predList = [pred for pred in predList if pred['score'] >= threshold]
    pred_img = draw_annotations_on_image(original_img, predList, color=(255, 0, 0), show_scores=True)
    axes[2].imshow(pred_img)
    axes[2].set_title(f'Predictions ({len(predList)} detections)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    savedFilename = os.path.join(output_dir, f"image_{img['id']:04d}.png")
    plt.savefig(savedFilename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savedFilename}")

## ===============used for temporary debugging=====================
def compute_ap_scores(coco_instance_result_file, annotation_file):
    """
    Compute AP (Average Precision) scores using COCO evaluation API.
    
    Returns:
        tuple: (ap_scores dict, coco_eval object for PR curve)
    """
    # Load ground truth
    coco_gt = COCO(annotation_file)
    
    # Load predictions
    with open(coco_instance_result_file, 'r') as f:
        coco_dt = coco_gt.loadRes(json.load(f))
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract AP scores
    ap_scores = {
        'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],    # AP @ IoU=0.50
        'AP75': coco_eval.stats[2],    # AP @ IoU=0.75
        'AP_small': coco_eval.stats[3], # AP for small objects
        'AP_medium': coco_eval.stats[4], # AP for medium objects
        'AP_large': coco_eval.stats[5],  # AP for large objects
        'AR_1': coco_eval.stats[6],     # AR given 1 det per image
        'AR_10': coco_eval.stats[7],   # AR given 10 dets per image
        'AR_100': coco_eval.stats[8],  # AR given 100 dets per image
    }
    
    return ap_scores, coco_eval

def print_ap_scores(ap_scores):
    """Print AP scores in a formatted way."""
    print("\n" + "=" * 60)
    print("COMPUTED AP SCORES")
    print("=" * 60)
    print(f"AP @ IoU=0.50:0.95:  {ap_scores['AP']:.4f}")
    print(f"AP @ IoU=0.50:       {ap_scores['AP50']:.4f}")
    print(f"AP @ IoU=0.75:       {ap_scores['AP75']:.4f}")
    print(f"\nAP by object size:")
    print(f"  AP_small:          {ap_scores['AP_small']:.4f}")
    print(f"  AP_medium:         {ap_scores['AP_medium']:.4f}")
    print(f"  AP_large:          {ap_scores['AP_large']:.4f}")
    print(f"\nAR (Average Recall):")
    print(f"  AR @ 1 det:        {ap_scores['AR_1']:.4f}")
    print(f"  AR @ 10 dets:       {ap_scores['AR_10']:.4f}")
    print(f"  AR @ 100 dets:      {ap_scores['AR_100']:.4f}")
    print("=" * 60 + "\n")
## ======================================================


def plot_pr_curve(coco_eval, output_path):
    """
    Plot Precision-Recall curve from COCO evaluation results.
    
    Args:
        coco_eval: COCOeval object after evaluation
        output_path: Path to save the PR curve image
    """
    # Get precision and recall arrays
    # coco_eval.eval['precision'] shape: [T, R, K, A, M]
    # T: IoU thresholds (0.5:0.05:0.95) = 10
    # R: recall thresholds (0:0.01:1) = 101
    # K: number of classes
    # A: number of area ranges
    # M: max dets per image
    
    precision = coco_eval.eval['precision']
    
    # Average over IoU thresholds (0.5:0.05:0.95), all classes, all area ranges, max dets
    # precision shape: [T=10, R=101, K=1, A=3, M=3]
    # We want to average over T, K, A, M to get precision vs recall curve
    pr = precision[:, :, 0, 0, 2]  # IoU=0.5:0.95, class 0, all areas, maxDets=100
    
    # Average over IoU thresholds to get single PR curve
    pr_mean = pr.mean(axis=0)  # Average over 10 IoU thresholds
    
    # Recall thresholds: 0:0.01:1 (101 points)
    recall = np.linspace(0, 1, 101)
    
    # Create PR curve plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot main PR curve (averaged over IoU thresholds)
    ax.plot(recall, pr_mean, 'b-', linewidth=2, label=f'AP={coco_eval.stats[0]:.4f}')
    
    # Also plot AP50 and AP75 curves for comparison
    pr_50 = precision[0, :, 0, 0, 2]  # IoU=0.50
    pr_75 = precision[5, :, 0, 0, 2]  # IoU=0.75 (index 5 = 0.75)
    
    ax.plot(recall, pr_50, 'g--', linewidth=1.5, alpha=0.7, label=f'AP50={coco_eval.stats[1]:.4f}')
    ax.plot(recall, pr_75, 'r--', linewidth=1.5, alpha=0.7, label=f'AP75={coco_eval.stats[2]:.4f}')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"PR curve saved to: {output_path}")
 
def analyze_predictions(coco_instance_result_file, image_path, annotation_file, output_dir, num_images, threshold):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


    # Compute AP scores first
    ap_scores, coco_eval = compute_ap_scores(coco_instance_result_file, annotation_file)

    with open(os.path.join(output_dir, "computed_ap_scores.txt"), 'w') as f:
        f.write(json.dumps(ap_scores, indent=4))
    print(f"AP scores saved to: {os.path.join(output_dir, 'computed_ap_scores.txt')}")

    
    # Plot PR curve
    pr_curve_path = os.path.join(output_dir, "pr_curve.png")
    plot_pr_curve(coco_eval, pr_curve_path)
    print(f"PR curve saved to: {pr_curve_path}")
    
    # take all distinct image_id in coco_instances_results.json
    predictionDict = {} ## image_id to prediction list

    with open(coco_instance_result_file, 'r') as f:
        predictions = json.load(f)
    for prediction in predictions:
        if prediction['image_id'] not in predictionDict:
            predictionDict[prediction['image_id']] = []
        predictionDict[prediction['image_id']].append(prediction)

    imageIDSet = set(predictionDict.keys())

    # Load all images and annotations first
    imageInfo = {} ## id to img info
    annotationDict = {} ## id to annotation list
    with open(annotation_file, 'r') as f:
        groundTruths = json.load(f)
        images = groundTruths['images']
        annotations = groundTruths['annotations']
        
        # Load all images into dictionary
        for image in images:
            imageInfo[image['id']] = image
        
        # Load all annotations
        for annotation in annotations:
            img_id = annotation['image_id']
            if img_id not in annotationDict:
                annotationDict[img_id] = []
            annotationDict[img_id].append(annotation)
    
    # Check which image IDs from predictions exist in ground truth
    gt_image_ids = set(imageInfo.keys())
    matching_ids = imageIDSet & gt_image_ids
    missing_ids = imageIDSet - gt_image_ids
    
    # Use only matching IDs
    imageIDSet = matching_ids
    
    # Save AP scores to file
    ap_scores_file = os.path.join(output_dir, "computed_ap_scores.txt")
    with open(ap_scores_file, 'w') as f:
        json.dump(ap_scores, f, indent=4)
    print(f"AP scores saved to: {ap_scores_file}")

    print_ap_scores(ap_scores)


    # Plot first 10 images
    for idx, imageID in enumerate(sorted(imageIDSet)[:num_images]):
        if imageID not in imageInfo:
            print(f"Image info not found for ID {imageID}")
            continue
        img = imageInfo[imageID]
        gt = annotationDict.get(imageID, [])
        pred = predictionDict.get(imageID, [])

        plot_gt_and_predictions(img, gt, pred, image_path, output_dir, threshold=threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_instance_result_file", type=str, default="output/inference/coco_instances_results.json")
    parser.add_argument("--image_path", type=str, default="../../coco_kvasirseg/val/images/")
    parser.add_argument("--annotation_file", type=str, default="../../coco_kvasirseg/val/annotations/val.json")
    
    parser.add_argument("--output_dir", type=str, default="evaluation/kvasirseg/visualizations/")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    analyze_predictions(args.coco_instance_result_file, 
                        args.image_path, 
                        args.annotation_file, 
                        args.output_dir, 
                        args.num_images, 
                        args.threshold)