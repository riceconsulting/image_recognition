# src/data/preprocess_masks.py
import os
import json
from PIL import Image
import numpy as np
import argparse

def preprocess_masks_for_dataset(dataset_name: str):
    """
    Converts binary ground truth masks into multi-class integer masks for a single dataset.
    It automatically discovers classes from the folder structure and saves a labels.json.
    """
    print(f"\n--- Processing dataset: {dataset_name} ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_dir = os.path.join(project_root, 'data', 'processed', dataset_name)
    
    # 1. Discover classes from the 'test' directory structure
    test_dir = os.path.join(dataset_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at '{test_dir}'. Skipping dataset.")
        return

    defect_types = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    # Create the class map: 'good' is always 0, defects start from 1
    class_map = {'good': 0}
    class_index = 1
    for defect in defect_types:
        if defect != 'good':
            class_map[defect] = class_index
            class_index += 1
            
    print(f"Generated class map: {class_map}")

    # Save the generated class map to a labels.json file
    labels_path = os.path.join(dataset_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(class_map, f, indent=2)
    print(f"Saved class map to {labels_path}")

    # 2. Define input and output directories
    source_dir = os.path.join(dataset_dir, 'ground_truth')
    output_dir = os.path.join(dataset_dir, 'ground_truth_multiclass')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading binary masks from: {source_dir}")
    print(f"Saving multi-class masks to: {output_dir}")

    # 3. Iterate through defect types and process masks
    processed_count = 0
    for defect_type, class_idx in class_map.items():
        if class_idx == 0: continue

        defect_folder = os.path.join(source_dir, defect_type)
        if not os.path.isdir(defect_folder):
            print(f"Warning: No ground_truth folder for defect '{defect_type}'. Skipping.")
            continue
            
        os.makedirs(os.path.join(output_dir, defect_type), exist_ok=True)

        for mask_filename in os.listdir(defect_folder):
            if mask_filename.endswith('_mask.png'):
                mask_path = os.path.join(defect_folder, mask_filename)
                binary_mask = Image.open(mask_path).convert('L')
                mask_array = np.array(binary_mask)

                multiclass_mask_array = np.zeros_like(mask_array, dtype=np.uint8)
                multiclass_mask_array[mask_array > 0] = class_idx

                output_mask_path = os.path.join(output_dir, defect_type, mask_filename)
                Image.fromarray(multiclass_mask_array).save(output_mask_path)
                processed_count += 1

    print(f"Preprocessing complete for '{dataset_name}'. Processed {processed_count} masks.")

if __name__ == '__main__':
    """
    Main execution block to automatically find and process all datasets
    in the data/processed directory.
    """
    print("--- Starting Automatic Mask Preprocessing ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_dir = os.path.join(project_root, 'data', 'processed')

    if not os.path.exists(processed_data_dir):
        print(f"Error: 'data/processed' directory not found at {processed_data_dir}")
    else:
        # Find all subdirectories, which are assumed to be datasets
        all_datasets = [d for d in os.listdir(processed_data_dir) if os.path.isdir(os.path.join(processed_data_dir, d))]
        
        if not all_datasets:
            print("No dataset folders found in 'data/processed'.")
        else:
            print(f"Found datasets: {', '.join(all_datasets)}")
            for dataset_name in all_datasets:
                preprocess_masks_for_dataset(dataset_name)
    
    print("\n--- Automation Finished ---")
