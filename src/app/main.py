# src/app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from . import api

# Initialize the FastAPI app
app = FastAPI(title="Industrial Defect Detection API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router ---
app.include_router(api.router)

# --- Static Files & Frontend ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
frontend_dir = os.path.join(project_root, "src", "frontend")

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/data", StaticFiles(directory=os.path.join(project_root, "data")), name="data")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page for the user interface."""
    with open(os.path.join(frontend_dir, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/datasets")
async def get_datasets():
    """Endpoint to list available datasets from the data/processed folder."""
    processed_dir = os.path.join(project_root, "data", "processed")
    try:
        if os.path.exists(processed_dir) and os.path.isdir(processed_dir):
            datasets = [name for name in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, name))]
            return JSONResponse(content={"datasets": datasets})
        else:
            return JSONResponse(content={"datasets": []}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/example-images/{dataset_name}")
async def get_example_images(dataset_name: str):
    """Endpoint to find example images and their class names."""
    examples = []
    test_dir = os.path.join(project_root, "data", "processed", dataset_name, "test")
    if not os.path.exists(test_dir):
        return JSONResponse(content={"error": "Test directory not found"}, status_code=404)

    # Find one image from each sub-directory in the test folder
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if os.path.isdir(defect_dir):
            for img_file in sorted(os.listdir(defect_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    examples.append({
                        "path": f"data/processed/{dataset_name}/test/{defect_type}/{img_file}",
                        "className": defect_type.replace('_', ' ').title()
                    })
                    break # Move to the next defect type
    
    return JSONResponse(content={"images": examples})
