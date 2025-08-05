# app/controller.py
from PIL import Image
from src.core.inference_engine import InferenceEngine
from src.train import run_training

# --- Configuration ---
# This would ideally be loaded from a centralized config file (e.g., config/config.yaml)
CONFIG = {
    'model_path': 'models/final/defect_detector_v1.pth',
    'model_config': {
        'num_classes': 2,
        'pretrained': False 
    },
    'training_config': {
        'train_dir': 'data/processed/train',
        'val_dir': 'data/processed/validation',
        'model_config': {
            'num_classes': 2,
            'pretrained': True
        },
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 25,
        'checkpoint_path': 'models/checkpoints/latest.pth',
        'final_model_path': 'models/final/defect_detector_v1.pth'
    }
}

# --- Singleton for Inference Engine ---
# This ensures the model is loaded only once.
inference_engine = None

def get_inference_engine():
    """Initializes and returns the inference engine."""
    global inference_engine
    if inference_engine is None:
        print("Initializing Inference Engine...")
        inference_engine = InferenceEngine(model_path=CONFIG['model_path'], config=CONFIG)
    return inference_engine

# --- Controller Functions ---

def analyze_image(image: Image.Image):
    """
    Analyzes an image to detect defects.
    
    Args:
        image (Image.Image): The image to analyze.
        
    Returns:
        dict: The prediction result.
    """
    engine = get_inference_engine()
    prediction = engine.predict(image)
    return prediction

def start_model_training():
    """
    Triggers the model training process.
    
    Returns:
        dict: A message indicating the training has started.
    """
    print("Controller received request to start training.")
    # In a production system, this would be an asynchronous task (e.g., using Celery)
    # For this baseline, we'll run it synchronously.
    run_training(CONFIG['training_config'])
    
    # After training, the inference engine is now out of date.
    # We set it to None so it gets re-initialized with the new model on the next predict call.
    global inference_engine
    inference_engine = None
    print("Training complete. Inference engine will be reloaded on next request.")

    return {"status": "success", "message": "Model training completed successfully. The new model is now active."}
