# src/app/controller.py
from PIL import Image
import base64
import io
from src.core.inference_engine import InferenceEngine
from src.train import run_training
from src.config_loader import get_model_config

# --- Engine Cache ---
# This dictionary will store loaded inference engines to avoid reloading them.
# Key: model_architecture (e.g., 'unet'), Value: InferenceEngine instance
inference_engines = {}

def get_inference_engine(model_architecture: str):
    """
    Initializes and returns the appropriate inference engine from a cache.
    If the engine is not cached, it loads the model and caches it.
    """
    if model_architecture not in inference_engines:
        print(f"Initializing Inference Engine for '{model_architecture}'...")
        config = get_model_config(model_architecture)
        inference_engines[model_architecture] = InferenceEngine(
            model_path=config['paths']['best_model'],
            config=config
        )
    return inference_engines[model_architecture]

def analyze_image(image: Image.Image, model_architecture: str):
    """
    Analyzes an image using the specified model architecture.
    """
    engine = get_inference_engine(model_architecture)
    prediction = engine.predict(image)
    
    # For segmentation, encode the overlay image to send via API
    buffered = io.BytesIO()
    prediction['overlay_image'].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "defect_detected": prediction['defect_detected'],
        "overlay_image_b64": img_str
    }

def start_model_training(model_architecture: str):
    """
    Triggers the training process for a specific model architecture.
    """
    print(f"Controller received request to start training for '{model_architecture}'.")
    run_training(model_architecture)
    
    # After training, remove the old engine from the cache so it gets reloaded
    # with the new model on the next predict request.
    if model_architecture in inference_engines:
        del inference_engines[model_architecture]
        print(f"Removed old '{model_architecture}' engine from cache. It will be reloaded on the next request.")

    return {"status": "success", "message": f"Model training for '{model_architecture}' completed."}
