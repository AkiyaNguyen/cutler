import os, shutil
from detectron2.data import DatasetCatalog, MetadataCatalog
        

def log_util(result, metrics_file = None):
    if metrics_file is None:
        raise ValueError("metrics_file is required")
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(str(result) + "\n\n")
        
        # Format metrics nicely
        if isinstance(result, dict):
            f.write("Detailed Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_file}")


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