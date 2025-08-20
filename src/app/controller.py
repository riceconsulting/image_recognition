# src/app/controller.py
from PIL import Image
import base64
import io
import os
import json
import datetime
from src.core.inference_engine import InferenceEngine
from src.train import run_training
from src.config_loader import get_model_config

inference_engines = {}

def get_inference_engine(model_architecture: str, dataset_name: str):
    cache_key = (model_architecture, dataset_name)
    if cache_key not in inference_engines:
        config = get_model_config(model_architecture, dataset_name)
        
        original_path = config['paths']['best_model']
        path_parts = original_path.rsplit('.', 1)
        model_path = f"{path_parts[0]}_{dataset_name}.{path_parts[1]}"

        inference_engines[cache_key] = InferenceEngine(model_path=model_path, config=config)
    return inference_engines[cache_key]

def log_prediction(prediction_data: dict, dataset_name: str, original_image: Image.Image):
    """
    Appends a prediction result to a JSON Lines log file.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_path = os.path.join(project_root, "prediction_log.jsonl")
    
    # Convert original image to a small thumbnail for the log
    thumbnail = original_image.copy()
    thumbnail.thumbnail((100, 100))
    buffered = io.BytesIO()
    thumbnail.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "dataset_name": dataset_name,
        "defects_found": prediction_data.get("defects_found", {}),
        "inference_time_ms": prediction_data.get("inference_time_ms", 0.0),
        "thumbnail_b64": img_str
    }
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def analyze_image(image: Image.Image, model_architecture: str, dataset_name: str):
    engine = get_inference_engine(model_architecture, dataset_name)
    prediction = engine.predict(image)
    
    # Log the prediction for the dashboard
    log_prediction(prediction, dataset_name, image)
    
    buffered = io.BytesIO()
    prediction['overlay_image'].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "defects_found": prediction.get("defects_found", {}),
        "overlay_image_b64": img_str,
        "inference_time_ms": prediction.get('inference_time_ms', 0.0)
    }

def start_model_training(model_architecture: str, dataset_name: str):
    history = run_training(model_architecture=model_architecture, dataset_name=dataset_name)
    
    cache_key = (model_architecture, dataset_name)
    if cache_key in inference_engines:
        del inference_engines[cache_key]

    return { "status": "success", "message": "Training completed.", "history": history }
