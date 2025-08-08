# src/app/api.py
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Query
from PIL import Image
import io
from . import controller
from typing import Literal

# Create a new router object
router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Welcome to the Defect Detection API."}

@router.post("/predict/")
async def predict(
    image_file: UploadFile = File(..., description="The image to analyze."),
    model_arch: Literal['unet', 'deeplabv3plus'] = Query('unet', description="The model architecture to use for prediction.")
):
    """
    Prediction endpoint. Receives an image and a model choice, returns the analysis.
    """
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    prediction = controller.analyze_image(image, model_architecture=model_arch)
    
    return prediction

@router.post("/train/")
async def train(
    background_tasks: BackgroundTasks,
    model_arch: Literal['unet', 'deeplabv3plus'] = Query('unet', description="The model architecture to train.")
):
    """
    Training endpoint. Triggers the training process for a specific model in the background.
    """
    background_tasks.add_task(controller.start_model_training, model_architecture=model_arch)
    return {"message": f"Model training for '{model_arch}' started in the background."}
