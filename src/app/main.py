# app/main.py
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from PIL import Image
import io

# --- FIX ---
# Import the specific functions needed from the controller module.
from .controller import get_inference_engine, analyze_image, start_model_training

# Initialize the FastAPI app
app = FastAPI(title="Industrial Defect Detection API")

@app.on_event("startup")
def startup_event():
    """
    On startup, this will initialize the inference engine so the first
    prediction is fast.
    """
    print("--- Application Startup ---")
    # Call the function directly without the 'controller.' prefix
    get_inference_engine()
    print("--- API is ready ---")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Defect Detection API. Use '/predict' to analyze an image or '/train' to start model training."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint. Receives an image file and returns the prediction.
    The logic is handled by the controller.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Call the function directly
    prediction = analyze_image(image)
    
    return prediction

@app.post("/train/")
async def train(background_tasks: BackgroundTasks):
    """
    Training endpoint. Triggers the model training process in the background.
    """
    # Call the function directly
    background_tasks.add_task(start_model_training)
    return {"message": "Model training started in the background. This may take some time. The new model will be active once training is complete."}
