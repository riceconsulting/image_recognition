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
from src.core.post_processing import process_multiclass_segmentation_output, overlay_multiclass_mask_on_image

class InferenceEngine:
    def __init__(self, model_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
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

        # --- FIX: Perform a warm-up run ---
        self._warmup()

    def _warmup(self):
        """
        Performs a single dummy inference to warm up the model, GPU, and any
        JIT compilers. This ensures the first real prediction has an accurate time.
        """
        print("Warming up the inference engine...")
        # Create a dummy tensor with the expected input shape [batch, channels, height, width]
        dummy_input = torch.randn(1, 3, 256, 256, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)
        print("Warm-up complete.")

    def predict(self, image: Image.Image):
        original_image = image.copy()
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Now the timing for the first prediction will be accurate
        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(image_tensor)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        if isinstance(output, dict): output = output['out']
            
        pred_mask = process_multiclass_segmentation_output(output)
        
        detected_defects = {}
        idx_to_class = {v: k for k, v in self.config.get('class_map', {}).items()}
        
        # --- FIX: Calculate total pixels for percentage calculation ---
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        
        unique_classes = np.unique(pred_mask)
        for class_idx in unique_classes:
            if class_idx == 0: continue
            
            class_name = idx_to_class.get(class_idx, f"Unknown_{class_idx}")
            pixel_count = np.sum(pred_mask == class_idx)
            
            # --- FIX: Calculate area percentage ---
            area_percentage = (pixel_count / total_pixels) * 100
            
            # --- FIX: Add area_percentage to the response dictionary ---
            detected_defects[class_name] = { 
                "area_pixels": int(pixel_count),
                "area_percentage": area_percentage
            }
            
        overlayed_image = overlay_multiclass_mask_on_image(original_image, pred_mask)
        
        return {
            "defects_found": detected_defects,
            "overlay_image": overlayed_image,
            "inference_time_ms": inference_time_ms,
        }
