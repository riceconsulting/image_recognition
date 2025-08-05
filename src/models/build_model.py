# src/models/build_model.py
from src.models.model_architectures import get_resnet_model

def build_defect_detection_model(config):
    """
    Builds the model based on a configuration dictionary.
    This makes it easy to change model parameters from a single config file.
    
    Args:
        config (dict): A dictionary containing model parameters like 'num_classes'.
    
    Returns:
        A PyTorch model instance.
    """
    num_classes = config.get('num_classes', 2) # Default to 2 classes
    pretrained = config.get('pretrained', True)
    
    print(f"Building model for {num_classes} classes...")
    model = get_resnet_model(num_classes=num_classes, pretrained=pretrained)
    
    return model
