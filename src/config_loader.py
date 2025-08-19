# src/config_loader.py
import yaml
import os
import json

def load_full_config():
    """Loads the entire configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_config(model_architecture: str, dataset_name: str):
    """
    Loads the configuration for a specific model and dynamically loads the
    class map for the specified dataset if it exists.
    """
    full_config = load_full_config()
    
    if model_architecture not in full_config:
        raise ValueError(f"Configuration for model '{model_architecture}' not found in config.yaml")
        
    config = full_config[model_architecture]
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    labels_path = os.path.join(project_root, 'data', 'processed', dataset_name, 'labels.json')
    
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_map = json.load(f)
        config['class_map'] = class_map
        config['model_config']['n_classes'] = len(class_map)
    
    config['product_name'] = dataset_name
    config['active_model_architecture'] = model_architecture
    
    print(f"--- Configuration loaded for model: '{model_architecture}' on dataset: '{dataset_name}' ---")
    return config
