# src/models/build_model.py
from src.models.model_architectures import UNet

def build_segmentation_model(config):
    """
    Builds the U-Net segmentation model.
    """
    n_channels = config.get('n_channels', 3)
    n_classes = config.get('n_classes', 1) # For binary segmentation (defect vs. background)

    print(f"Building U-Net model for {n_classes} output class(es)...")
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    
    return model
