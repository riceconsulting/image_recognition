# src/core/post_processing.py
import torch
import numpy as np
from PIL import Image

# Define a flexible color map. More colors can be added.
COLOR_MAP = [
    (0, 0, 0),       # 0: background (Black)
    (255, 0, 0),     # 1: class 1 (Red)
    (0, 255, 0),     # 2: class 2 (Green)
    (0, 0, 255),     # 3: class 3 (Blue)
    (255, 255, 0),   # 4: class 4 (Yellow)
    (255, 0, 255),   # 5: class 5 (Magenta)
]

def process_multiclass_segmentation_output(model_output):
    """
    Processes raw multi-class output into a class index mask.
    """
    pred_mask = torch.argmax(torch.softmax(model_output, dim=1), dim=1)
    return pred_mask.cpu().numpy().squeeze().astype(np.uint8)

def overlay_multiclass_mask_on_image(image: Image.Image, mask: np.ndarray, alpha=0.5):
    """
    Overlays a multi-class segmentation mask on an image using a color map.
    """
    color_overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_idx in np.unique(mask):
        if class_idx == 0: continue # Skip background
        
        # --- FIX ---
        # Use the modulo operator (%) to prevent IndexError if there are more
        # classes than colors defined in the COLOR_MAP.
        color = COLOR_MAP[class_idx % len(COLOR_MAP)]
        
        color_overlay[mask == class_idx] = color

    overlay_img = Image.fromarray(color_overlay)
    
    if overlay_img.size != image.size:
        overlay_img = overlay_img.resize(image.size, resample=Image.NEAREST)

    final_image = Image.blend(image.convert('RGB'), overlay_img, alpha)
    return final_image
