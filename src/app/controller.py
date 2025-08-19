# src/app/controller.py
from PIL import Image
import base64
import io
from src.core.inference_engine import InferenceEngine
from src.train import run_training
from src.config_loader import get_model_config

inference_engines = {}

def get_inference_engine(model_architecture: str, dataset_name: str):
    cache_key = (model_architecture, dataset_name)
    if cache_key not in inference_engines:
        print(f"Initializing Inference Engine for '{model_architecture}' on dataset '{dataset_name}'...")
        config = get_model_config(model_architecture, dataset_name)
        
        original_path = config['paths']['best_model']
        path_parts = original_path.rsplit('.', 1)
        model_path = f"{path_parts[0]}_{dataset_name}.{path_parts[1]}"

        inference_engines[cache_key] = InferenceEngine(
            model_path=model_path,
            config=config
        )
    return inference_engines[cache_key]

def analyze_image(image: Image.Image, model_architecture: str, dataset_name: str):
    engine = get_inference_engine(model_architecture, dataset_name)
    prediction = engine.predict(image)
    
    buffered = io.BytesIO()
    prediction['overlay_image'].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "defect_detected": prediction['defect_detected'],
        "overlay_image_b64": img_str,
        "confidence": prediction.get('confidence', 0.0),
        "inference_time_ms": prediction.get('inference_time_ms', 0.0)
    }

def start_model_training(model_architecture: str, dataset_name: str):
    history = run_training(model_architecture=model_architecture, dataset_name=dataset_name)
    
    cache_key = (model_architecture, dataset_name)
    if cache_key in inference_engines:
        del inference_engines[cache_key]

    return {
        "status": "success", 
        "message": "Training completed.",
        "history": history
    }
