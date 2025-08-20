# src/config_loader.py
import yaml
import os
import json

def load_full_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_config(model_architecture: str, dataset_name: str):
    full_config = load_full_config()
    
    if model_architecture not in full_config:
        raise ValueError(f"Config for model '{model_architecture}' not found.")
        
    config = full_config[model_architecture]
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    labels_path = os.path.join(project_root, 'data', 'processed', dataset_name, 'labels.json')
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found for '{dataset_name}'. Please run preprocess_masks.py first.")
        
    with open(labels_path, 'r') as f:
        class_map = json.load(f)
    
    config['class_map'] = class_map
    config['model_config']['n_classes'] = len(class_map)
    config['product_name'] = dataset_name
    config['active_model_architecture'] = model_architecture
    
    print(f"--- Config loaded for model: '{model_architecture}' on dataset: '{dataset_name}' ---")
    print(f"Found {len(class_map)} classes: {list(class_map.keys())}")
    return config
