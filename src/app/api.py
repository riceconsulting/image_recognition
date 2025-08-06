# src/app/api.py
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from PIL import Image
import io
from .controller import analyze_image, start_model_training

# Create a new router object. All endpoints will be attached to this.
router = APIRouter()

@router.get("/")
def read_root():
    """Defines the root endpoint."""
    return {"message": "Welcome to the Defect Detection API. Use '/predict' to analyze an image or '/train' to start model training."}

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint. Receives an image file and returns the prediction.
    The logic is handled by the controller.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    prediction = analyze_image(image)
    
    return prediction

@router.post("/train/")
async def train(background_tasks: BackgroundTasks):
    """
    Training endpoint. Triggers the model training process in the background.
    The logic is handled by the controller.
    """
    background_tasks.add_task(start_model_training)
    return {"message": "Model training started in the background. This may take some time. The new model will be active once training is complete."}
