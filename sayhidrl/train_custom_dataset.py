#!/usr/bin/env python
"""
Complete example script for training CutLER on a custom dataset.

This script:
1. Registers your custom dataset
2. Sets up the training configuration
3. Runs training

Usage:
    python train_custom_dataset.py \
        --dataset-root /path/to/your/dataset \
        --train-images images/train \
        --train-json annotations/train.json \
        --val-images images/val \
        --val-json annotations/val.json \
        --num-gpus 8 \
        --output-dir output/my_training
"""

import argparse
import os
import sys
import importlib.util

# Add the cutler package to path
# CutLER library is located at /workspace/fbcutler
cutler_root = "/workspace/fbcutler"
sys.path.insert(0, cutler_root)

from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Import register_coco_instances directly without triggering cutler/__init__.py
coco_module_path = os.path.join(cutler_root, "cutler", "data", "datasets", "coco.py")
spec = importlib.util.spec_from_file_location("coco_module", coco_module_path)
coco_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coco_module)
register_coco_instances = coco_module.register_coco_instances

# Import Trainer, setup, and default_argument_parser from train_net
# We need to change to the cutler directory so relative imports work
cutler_package_path = os.path.join(cutler_root, "cutler")
original_dir = os.getcwd()
original_path = sys.path[:]

try:
    # Change to cutler directory and add it to path
    # This allows train_net.py's relative imports (from config, from engine) to work
    os.chdir(cutler_package_path)
    sys.path.insert(0, cutler_package_path)
    
    # Now import train_net - relative imports will work
    import train_net  # type: ignore  # Dynamic import
    Trainer = train_net.Trainer
    setup = train_net.setup
    default_argument_parser = train_net.default_argument_parser
finally:
    # Restore original directory and path
    os.chdir(original_dir)
    sys.path[:] = original_path

def register_dataset(dataset_name, images_path, json_path, metadata=None):
    """Register a COCO-format dataset."""
    if metadata is None:
        metadata = {}
    
    register_coco_instances(
        dataset_name,
        metadata,
        json_path,
        images_path,
    )
    print(f"Successfully registered dataset: {dataset_name}")
    print(f"  Images: {images_path}")
    print(f"  Annotations: {json_path}")


def main(args):
    # Register custom datasets
    train_dataset_name = args.train_dataset_name or "custom_train"
    val_dataset_name = args.val_dataset_name or "custom_val"
    
    # Build full paths
    if args.train_images and args.train_json:
        train_images = os.path.join(args.dataset_root, args.train_images)
        train_json = os.path.join(args.dataset_root, args.train_json)
        # Register training dataset
        register_dataset(train_dataset_name, train_images, train_json)
    
    # Build validation dataset paths
    val_images = os.path.join(args.dataset_root, args.val_images) if args.val_images else None
    val_json = os.path.join(args.dataset_root, args.val_json) if args.val_json else None
    
    # Register validation dataset
    if args.eval_only:
        # For eval-only mode, validation dataset is required
        if not args.val_images or not args.val_json:
            raise ValueError("--val-images and --val-json are required when using --eval-only")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Validation JSON file not found: {val_json}")
        if not os.path.exists(val_images):
            raise FileNotFoundError(f"Validation images directory not found: {val_images}")
        register_dataset(val_dataset_name, val_images, val_json)
        
        # Limit test dataset size if specified (for quick testing)
        if args.num_test_images and args.num_test_images > 0:
            from detectron2.data import DatasetCatalog, MetadataCatalog
            
            # Get the full dataset and limit it
            full_dataset = DatasetCatalog.get(val_dataset_name)
            limited_dataset = full_dataset[:args.num_test_images]
            
            # Register a limited version
            limited_dataset_name = val_dataset_name + "_limited"
            DatasetCatalog.register(limited_dataset_name, lambda: limited_dataset)
            
            # Copy metadata from original dataset to limited dataset
            original_metadata = MetadataCatalog.get(val_dataset_name)
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
            
            test_dataset = limited_dataset_name
            print(f"Limited test dataset to {len(limited_dataset)} images (from {len(full_dataset)} total)")
        else:
            test_dataset = val_dataset_name
    elif val_json and val_images and os.path.exists(val_json):
        # Validation dataset provided and exists
        register_dataset(val_dataset_name, val_images, val_json)
        test_dataset = val_dataset_name
    elif args.train_images and args.train_json:
        # Fallback to training dataset if validation not provided (training mode)
        test_dataset = train_dataset_name
    else:
        raise ValueError("Either training or validation dataset must be provided")
    
    # Set dataset names using the dedicated arguments (train_net.py handles these specially)
    if args.train_images and args.train_json:
        args.train_dataset = train_dataset_name
    args.test_dataset = test_dataset
    
    # Modify args.opts if needed
    if not hasattr(args, 'opts'):
        args.opts = []
    
    # Add weights if not already specified
    # Check if checkpoint argument was provided
    if args.checkpoint:
        weights_set = False
        for i in range(len(args.opts)):
            if args.opts[i] == 'MODEL.WEIGHTS':
                weights_set = True
                args.opts[i + 1] = args.checkpoint
                break
        if not weights_set:
            args.opts.extend(['MODEL.WEIGHTS', args.checkpoint])
    else:
        # Use default weights if not specified via checkpoint or opts
        weights_set = False
        for i in range(len(args.opts)):
            if args.opts[i] == 'MODEL.WEIGHTS':
                weights_set = True
                break
        if not weights_set:
            args.opts.extend(['MODEL.WEIGHTS', 'http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_r2.pth'])
    
    # Add output directory if specified
    output_dir_set = False
    for i in range(len(args.opts)):
        if args.opts[i] == 'OUTPUT_DIR':
            output_dir_set = True
            break
    
    if args.output_dir and not output_dir_set:
        args.opts.extend(['OUTPUT_DIR', args.output_dir])
    
    # Handle CPU mode
    if args.cpu:
        device_set = False
        for i in range(len(args.opts)):
            if args.opts[i] == 'MODEL.DEVICE':
                device_set = True
                args.opts[i + 1] = 'cpu'
                break
        if not device_set:
            args.opts.extend(['MODEL.DEVICE', 'cpu'])
     
    # Setup and run training
    cfg = setup(args)
    
    if args.eval_only:
        from detectron2.checkpoint import DetectionCheckpointer
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        
        # Save metrics to file
        metrics_file = os.path.join(cfg.OUTPUT_DIR, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(str(res) + "\n\n")
            
            # Format metrics nicely
            if isinstance(res, dict):
                f.write("Detailed Metrics:\n")
                f.write("-"*60 + "\n")
                for key, value in res.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        print(f"\nMetrics saved to: {metrics_file}")
        
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CutLER on custom dataset",
        parents=[default_argument_parser()],
        add_help=False,
    )   
    
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing your dataset",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        required=False,
        help="Path to training images (relative to dataset-root). Required for training, optional for eval-only",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=False,
        help="Path to training annotations JSON (relative to dataset-root). Required for training, optional for eval-only",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default=None,
        help="Path to validation images (relative to dataset-root). Required for eval-only, optional for training",
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default=None,
        help="Path to validation annotations JSON (relative to dataset-root). Required for eval-only, optional for training",
    )
    parser.add_argument(
        "--train-dataset-name",
        type=str,
        default="custom_train",
        help="Name for the registered training dataset",
    )
    parser.add_argument(
        "--val-dataset-name",
        type=str,
        default="custom_val",
        help="Name for the registered validation dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for training results",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU instead of GPU (sets MODEL.DEVICE to 'cpu')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint/weights file (overrides MODEL.WEIGHTS in config)",
    )
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=None,
        help="Limit the number of test images (useful for quick testing). Only applies to eval-only mode.",
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.eval_only:
        # For eval-only mode, validation dataset is required
        if not args.val_images or not args.val_json:
            parser.error("--val-images and --val-json are required when using --eval-only")
    else:
        # For training mode, training dataset is required
        if not args.train_images or not args.train_json:
            parser.error("--train-images and --train-json are required for training (without --eval-only)")
    
    # Ensure config file is provided
    if not args.config_file:
        # Use the custom config by default
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cascade_mask_rcnn_R_50_FPN_custom.yaml"
        )
        if os.path.exists(config_path):
            args.config_file = os.path.abspath(config_path)  # Use absolute path
        else:
            # Fallback to demo config in cutler
            args.config_file = os.path.abspath(os.path.join(
                cutler_root, 
                "cutler", 
                "model_zoo", 
                "configs", 
                "CutLER-ImageNet", 
                "cascade_mask_rcnn_R_50_FPN_demo.yaml"
            ))
    else:
        # Ensure config file path is absolute
        if not os.path.isabs(args.config_file):
            args.config_file = os.path.abspath(args.config_file)
    
    print("=" * 60)
    if args.eval_only:
        print("CutLER Validation Inference")
    else:
        print("CutLER Custom Dataset Training")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Training images: {args.train_images}")
    print(f"Training annotations: {args.train_json}")
    print(f"Validation images: {args.val_images}")
    print(f"Validation annotations: {args.val_json}")
    print(f"Config file: {args.config_file}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

