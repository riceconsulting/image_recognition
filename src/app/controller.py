# src/app/controller.py
from PIL import Image
import base64
import io
from src.core.inference_engine import InferenceEngine
from src.train import run_training

CONFIG = {
    'product_name': 'bottle',
    'model_config': {'n_classes': 1, 'n_channels': 3},
    'best_model_path': 'models/final/best_segmentation_model.pth',
    
    'training_config': {
        'product_name': 'bottle',
        'model_config': {'n_classes': 1, 'n_channels': 3},
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 50,
        'patience': 10,
        'best_model_path': 'models/final/best_segmentation_model.pth'
    }
}

inference_engine = None

def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        print("Initializing Segmentation Inference Engine...")
        inference_engine = InferenceEngine(
            model_path=CONFIG['best_model_path'], 
            config=CONFIG
        )
    return inference_engine

def analyze_image(image: Image.Image):
    engine = get_inference_engine()
    prediction = engine.predict(image)
    
    # For a real API, you'd send the image back, e.g., as base64
    buffered = io.BytesIO()
    prediction['overlay_image'].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # We remove the actual image and mask from the JSON response for brevity
    return {
        "defect_detected": prediction['defect_detected'],
        "overlay_image_b64": img_str
    }

def start_model_training():
    print("Controller received request to start training.")
    run_training(CONFIG['training_config'])
    
    global inference_engine
    inference_engine = None
    print("Training complete. Inference engine will be reloaded on next request.")

    return {"status": "success", "message": "Model training completed successfully."}
