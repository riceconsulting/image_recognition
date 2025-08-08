# src/core/inference_engine.py
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
sys.path.insert(0, project_root)

from src.models.build_model import build_model
from src.core.post_processing import process_segmentation_output, overlay_mask_on_image

class InferenceEngine:
    def __init__(self, model_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = build_model(config['model_config'])
        
        print(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Simple transform for inference: resize, to tensor, normalize
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image):
        original_image = image.copy()
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)

        # --- FIX ---
        # Handle the dictionary output from torchvision's DeepLabV3+ model
        if isinstance(output, dict):
            output = output['out']
            
        binary_mask = process_segmentation_output(output)
        
        # Check if any defect was detected
        defect_detected = bool(np.sum(binary_mask) > 0)
        
        # Create an image with the mask overlaid for visualization
        overlayed_image = overlay_mask_on_image(original_image, binary_mask)
        
        return {
            "defect_detected": defect_detected,
            "mask": binary_mask.tolist(), # Send mask as a list
            "overlay_image": overlayed_image # This would be converted to base64 to send via API
        }
