#!/usr/bin/env python
"""
Fix COCO JSON files to include required 'info' and 'licenses' fields.
"""

import json
import sys
import os

def fix_coco_json(json_path):
    """Add missing 'info' and 'licenses' fields to COCO JSON file."""
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return False
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Add 'info' field if missing
    if 'info' not in data:
        data['info'] = {
            "description": "Custom dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-01-01"
        }
        print(f"Added 'info' field to {json_path}")
    
    # Add 'licenses' field if missing
    if 'licenses' not in data:
        data['licenses'] = [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ]
        print(f"Added 'licenses' field to {json_path}")
    
    # Backup original file
    backup_path = json_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Created backup: {backup_path}")
    
    # Write fixed JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully fixed {json_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_coco_json.py <json_file_path>")
        print("Example: python fix_coco_json.py ../coco_kvasirseg/eval/annotations/val.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    fix_coco_json(json_path)

