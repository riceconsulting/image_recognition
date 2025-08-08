# src/config_loader.py
import yaml
import os

def load_full_config():
    """Loads the entire configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_config(model_architecture: str):
    """
    Loads the configuration for a specific model architecture.
    """
    full_config = load_full_config()
    
    if model_architecture not in full_config:
        raise ValueError(f"Configuration for model '{model_architecture}' not found in config.yaml")
        
    config = full_config[model_architecture]
    
    # Add general settings to the config
    config['product_name'] = full_config.get('product_name', 'default_product')
    config['active_model_architecture'] = model_architecture
    
    print(f"--- Configuration loaded for model: '{model_architecture}' ---")
    return config
