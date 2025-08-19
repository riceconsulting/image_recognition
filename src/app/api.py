# src/app/api.py
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Query
from PIL import Image
import io
from . import controller
from typing import Literal

router = APIRouter()

@router.post("/predict/")
async def predict(
    image_file: UploadFile = File(..., description="The image to analyze."),
    model_arch: Literal['unet', 'deeplabv3plus'] = Query('unet', description="The model architecture to use."),
    dataset_name: str = Query(..., description="The dataset the model was trained on (e.g., 'bottle').")
):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    prediction = controller.analyze_image(image, model_architecture=model_arch, dataset_name=dataset_name)
    return prediction

@router.post("/train/")
async def train(
    model_arch: Literal['unet', 'deeplabv3plus'] = Query('unet', description="The model architecture to train."),
    dataset_name: str = Query(..., description="The dataset to train the model on (e.g., 'bottle').")
):
    """
    Training endpoint that now runs synchronously and returns training history.
    """
    # Note: For long training jobs, a background task is better, but for this UI,
    # a synchronous response is needed to get the history back.
    response_data = controller.start_model_training(model_architecture=model_arch, dataset_name=dataset_name)
    return response_data
