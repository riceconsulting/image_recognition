# src/core/inference_engine.py
import torch
from PIL import Image
from src.models.build_model import build_defect_detection_model
from src.data.augmentation import get_validation_transforms

class InferenceEngine:
    """
    A class to handle loading the model and running inference.
    """
    def __init__(self, model_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build the model architecture
        self.model = build_defect_detection_model(config['model_config'])
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Set the model to evaluation mode
        
        self.transform = get_validation_transforms()
        self.class_names = ['good', 'defect'] # Should match your training labels

    def predict(self, image: Image.Image):
        """
        Takes a PIL image and returns a prediction.
        """
        # Pre-process the image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        # Post-process the output
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        
        predicted_class_name = self.class_names[predicted_class_idx.item()]
        
        return {
            "class_name": predicted_class_name,
            "confidence": confidence.item()
        }

# --- Example Usage (not part of the final app) ---
if __name__ == '__main__':
    # This is how you would initialize and use the engine
    CONFIG = {
        'model_config': {'num_classes': 2, 'pretrained': False} # pretrained doesn't matter for loading
    }
    engine = InferenceEngine(model_path='models/final/defect_detector_v1.pth', config=CONFIG)
    
    # Create a dummy image to test
    dummy_image = Image.new('RGB', (224, 224), color = 'red')
    
    prediction = engine.predict(dummy_image)
    print(f"Prediction: {prediction}")
