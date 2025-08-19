# src/core/inference_engine.py
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import time

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

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image):
        original_image = image.copy()
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # --- 1. Measure Inference Speed ---
        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(image_tensor)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        if isinstance(output, dict):
            output = output['out']
            
        # --- 2. Calculate Confidence Score ---
        probabilities = torch.sigmoid(output)
        binary_mask_tensor = (probabilities > 0.5)
        
        # Calculate confidence only for the pixels predicted as defects
        defect_pixels = probabilities[binary_mask_tensor]
        confidence_score = defect_pixels.mean().item() if len(defect_pixels) > 0 else 0.0

        # Convert to numpy for post-processing
        binary_mask = binary_mask_tensor.cpu().numpy().squeeze().astype(np.uint8)
        
        defect_detected = bool(np.sum(binary_mask) > 0)
        
        overlayed_image = overlay_mask_on_image(original_image, binary_mask)
        
        return {
            "defect_detected": defect_detected,
            "overlay_image": overlayed_image,
            "confidence": confidence_score,
            "inference_time_ms": inference_time_ms,
            # Note: Metrics like IoU/Dice require a ground truth mask,
            # which is not available during live prediction.
            # They are calculated in the evaluate.py script.
        }
