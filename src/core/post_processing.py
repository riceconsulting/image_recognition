# src/core/post_processing.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def process_segmentation_output(model_output, threshold=0.5):
    """
    Processes the raw output from a segmentation model into a binary mask.

    Args:
        model_output (torch.Tensor): The raw logits from the model.
        threshold (float): The threshold to binarize the probabilities.

    Returns:
        np.ndarray: A binary (0 or 1) numpy array representing the defect mask.
    """
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(model_output)
    
    # Binarize the output based on the threshold
    binary_mask = (probabilities > threshold).cpu().numpy().squeeze().astype(np.uint8)
    
    return binary_mask

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a segmentation mask on an image.

    Args:
        image (PIL.Image.Image): The original input image.
        mask (np.ndarray): The binary mask (0s and 1s).
        color (tuple): The RGB color for the mask overlay.
        alpha (float): The transparency of the overlay.

    Returns:
        PIL.Image.Image: The image with the mask overlaid.
    """
    # Convert mask to a PIL Image that can be used as a transparency layer
    mask_img = Image.fromarray(mask * 255, 'L')

    # --- FIX ---
    # Resize the mask to match the original image's size
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, resample=Image.NEAREST)

    # Create a color overlay from the mask
    overlay = Image.new('RGB', image.size, color)
    
    # Blend the original image with the overlay using the resized mask
    image_with_overlay = Image.composite(overlay, image, mask_img)
    
    # Final blend for transparency
    final_image = Image.blend(image, image_with_overlay, alpha)

    return final_image
