import argparse
import os
import sys
import importlib.util

## add cutler to path
script_dir = os.path.dirname(os.path.abspath(__file__))
cutler_root = os.path.join(script_dir, "..", "..", "fbcutler", 'cutler')
print(cutler_root)
sys.path.insert(0, cutler_root)

from detectron2.data import MetadataCatalog
from detectron2.engine import launch

from data.datasets import register_coco_instances #type: ignore
from train_net import Trainer, setup, default_argument_parser #type: ignore
from helper_func import extract_first_images
def register_dataset(dataset_name, images_path, json_path, metadata=None):
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
    train_dataset_name = args.train_dataset if args.train_dataset else "custom_train"
    val_dataset_name = args.test_dataset if args.test_dataset else "custom_val"
    
    if args.train_images and args.train_json:
        train_images = os.path.join(args.dataset_root, args.train_images)
        train_json = os.path.join(args.dataset_root, args.train_json)
        # Register training dataset
        register_dataset(train_dataset_name, train_images, train_json)
    
    # Build validation dataset paths
    val_images = os.path.join(args.dataset_root, args.val_images) if args.val_images else None
    val_json = os.path.join(args.dataset_root, args.val_json) if args.val_json else None
    
    if args.eval_only:
        if not args.val_images or not args.val_json:
            raise ValueError("--val-images and --val-json are required when using --eval-only")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Validation JSON file not found: {val_json}")
        if not os.path.exists(val_images):
            raise FileNotFoundError(f"Validation images directory not found: {val_images}")
        register_dataset(val_dataset_name, val_images, val_json)
    elif val_json and val_images:
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Validation JSON file not found: {val_json}")
        if not os.path.exists(val_images):
            raise FileNotFoundError(f"Validation images directory not found: {val_images}")
        register_dataset(val_dataset_name, val_images, val_json)
    else:
        raise ValueError("Either training or validation dataset must be provided")
    
    args.train_dataset = train_dataset_name
    args.test_dataset = val_dataset_name

    if args.test_code:
        test_number = args.num_test_images
        train_dataset_name = extract_first_images(train_dataset_name, test_number)
        val_dataset_name = extract_first_images(val_dataset_name, test_number)
        args.train_dataset = train_dataset_name
        args.test_dataset = val_dataset_name
     

    # Modify args.opts if needed
    if not hasattr(args, 'opts'):
        args.opts = []

    # Add output directory if specified
    output_dir_set = False
    for i in range(len(args.opts)):
        if args.opts[i] == 'OUTPUT_DIR':
            output_dir_set = True
            break
    if args.output_dir and not output_dir_set:
        args.opts.extend(['OUTPUT_DIR', args.output_dir])
    
    if args.cpu:
        assert args.eval_only, "CPU mode is only supported for evaluation"
        device_set = False
        for i in range(len(args.opts)):
            if args.opts[i] == 'MODEL.DEVICE':
                device_set = True
                args.opts[i + 1] = 'cpu'
                break
        if not device_set:
            args.opts.extend(['MODEL.DEVICE', 'cpu'])

    cfg = setup(args)
    if args.eval_only:  
        from detectron2.checkpoint import DetectionCheckpointer
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res;

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume = args.resume)
    return trainer.train()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Train CutLER on custom dataset",
        parents=[default_argument_parser()],
        add_help=False,
    )  
    parser.add_argument("--dataset-root", type=str, default="../../coco_kvasirseg")
    parser.add_argument("--train-images", type=str, default="train/images")
    parser.add_argument("--train-json", type=str, default="train/annotations/train.json")
    parser.add_argument("--val-images", type=str, default="val/images")
    parser.add_argument("--val-json", type=str, default="val/annotations/val.json")
    parser.add_argument("--output-dir", type=str, default="../output/my_training") ## relative to this script
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--test-code", action='store_true')
    parser.add_argument("--num-test-images", type=int, default=10)
    
    args = parser.parse_args()
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

    print("=" * 60)


    main(args)
    print("Training completed")