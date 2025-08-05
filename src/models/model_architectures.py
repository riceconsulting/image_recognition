# src/models/model_architectures.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes, pretrained=True):
    """
    Loads a pre-trained ResNet-18 model from torchvision.
    
    Args:
        num_classes (int): The number of output classes (e.g., 2 for 'good' vs 'defect').
        pretrained (bool): Whether to use weights pre-trained on ImageNet.
    
    Returns:
        A PyTorch model.
    """
    # Load a pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze all the parameters in the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer (the classifier)
    # with a new one for our specific number of classes.
    # The new layer's parameters will be trainable by default.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
