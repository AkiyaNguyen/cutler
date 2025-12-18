#!/usr/bin/env python
"""
Extract and display evaluation metrics from inference results.

Usage:
    python extract_metrics.py \
        --predictions output/kvasirseg_training/inference/coco_instances_results.json \
        --ground-truth ../coco_kvasirseg/eval/annotations/val.json \
        --output-file output/metrics.txt
"""

import argparse
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
cutler_root = os.path.join(script_dir, "..", "fbcutler")
sys.path.insert(0, cutler_root)

def evaluate_predictions(pred_file, gt_file, output_file=None):
    """
    Evaluate predictions against ground truth and print metrics.
    
    Args:
        pred_file: Path to predictions JSON file (COCO format)
        gt_file: Path to ground truth JSON file (COCO format)
        output_file: Optional file to save metrics
    """
    # Load ground truth
    print("Loading ground truth...")
    coco_gt = COCO(gt_file)
    
    # Load predictions
    print("Loading predictions...")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Ground truth: {len(coco_gt.getImgIds())} images")
    print(f"Predictions: {len(predictions)} detections")
    
    # Create COCO result object
    coco_dt = coco_gt.loadRes(pred_file)
    
    # Run evaluation for bounding boxes
    print("\n" + "="*60)
    print("Bounding Box Detection Metrics")
    print("="*60)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    bbox_metrics = {
        'AP': coco_eval_bbox.stats[0],
        'AP50': coco_eval_bbox.stats[1],
        'AP75': coco_eval_bbox.stats[2],
        'APs': coco_eval_bbox.stats[3],
        'APm': coco_eval_bbox.stats[4],
        'APl': coco_eval_bbox.stats[5],
        'AR1': coco_eval_bbox.stats[6],
        'AR10': coco_eval_bbox.stats[7],
        'AR100': coco_eval_bbox.stats[8],
        'ARs': coco_eval_bbox.stats[9],
        'ARm': coco_eval_bbox.stats[10],
        'ARl': coco_eval_bbox.stats[11],
    }
    
    # Run evaluation for segmentation
    print("\n" + "="*60)
    print("Segmentation Metrics")
    print("="*60)
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    segm_metrics = {
        'AP': coco_eval_segm.stats[0],
        'AP50': coco_eval_segm.stats[1],
        'AP75': coco_eval_segm.stats[2],
        'APs': coco_eval_segm.stats[3],
        'APm': coco_eval_segm.stats[4],
        'APl': coco_eval_segm.stats[5],
        'AR1': coco_eval_segm.stats[6],
        'AR10': coco_eval_segm.stats[7],
        'AR100': coco_eval_segm.stats[8],
        'ARs': coco_eval_segm.stats[9],
        'ARm': coco_eval_segm.stats[10],
        'ARl': coco_eval_segm.stats[11],
    }
    
    # Print formatted metrics
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)
    
    print("\nBounding Box Detection:")
    print(f"  AP (Average Precision):     {bbox_metrics['AP']:.4f}")
    print(f"  AP@0.50:                    {bbox_metrics['AP50']:.4f}")
    print(f"  AP@0.75:                    {bbox_metrics['AP75']:.4f}")
    print(f"  AP (small):                  {bbox_metrics['APs']:.4f}")
    print(f"  AP (medium):                 {bbox_metrics['APm']:.4f}")
    print(f"  AP (large):                  {bbox_metrics['APl']:.4f}")
    print(f"  AR@1:                        {bbox_metrics['AR1']:.4f}")
    print(f"  AR@10:                       {bbox_metrics['AR10']:.4f}")
    print(f"  AR@100:                      {bbox_metrics['AR100']:.4f}")
    
    print("\nSegmentation:")
    print(f"  AP (Average Precision):     {segm_metrics['AP']:.4f}")
    print(f"  AP@0.50:                     {segm_metrics['AP50']:.4f}")
    print(f"  AP@0.75:                     {segm_metrics['AP75']:.4f}")
    print(f"  AP (small):                  {segm_metrics['APs']:.4f}")
    print(f"  AP (medium):                 {segm_metrics['APm']:.4f}")
    print(f"  AP (large):                  {segm_metrics['APl']:.4f}")
    print(f"  AR@1:                        {segm_metrics['AR1']:.4f}")
    print(f"  AR@10:                       {segm_metrics['AR10']:.4f}")
    print(f"  AR@100:                      {segm_metrics['AR100']:.4f}")
    
    # Save to file if specified
    if output_file:
        metrics_dict = {
            'bbox': bbox_metrics,
            'segm': segm_metrics
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        metrics_dict = convert_numpy(metrics_dict)
        
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nMetrics saved to: {output_file}")
        
        # Also save human-readable text version
        txt_file = output_file.replace('.json', '.txt') if output_file.endswith('.json') else output_file + '.txt'
        with open(txt_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Bounding Box Detection:\n")
            for key, value in bbox_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nSegmentation:\n")
            for key, value in segm_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        print(f"Human-readable metrics saved to: {txt_file}")
    
    return bbox_metrics, segm_metrics


def main(args):
    evaluate_predictions(args.predictions, args.ground_truth, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display evaluation metrics")
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
        "--output-file",
        type=str,
        default=None,
        help="Optional file to save metrics (JSON and TXT formats)",
    )
    
    args = parser.parse_args()
    main(args)

