# src/models/build_model.py
from .model_architectures import UNet, get_deeplabv3plus

def build_model(config):
    """
    Builds the appropriate segmentation model based on the configuration.
    
    Args:
        config (dict): The model-specific configuration dictionary.
    
    Returns:
        A PyTorch model instance.
    """
    architecture = config.get('architecture')
    
    if architecture == 'unet':
        print("Building Segmentation Model (U-Net)...")
        return UNet(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 1)
        )
    elif architecture == 'deeplabv3plus':
        print("Building Segmentation Model (DeepLabV3+)...")
        return get_deeplabv3plus(
            num_classes=config.get('num_classes', 1),
            output_stride=config.get('output_stride', 16),
            pretrained=config.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
