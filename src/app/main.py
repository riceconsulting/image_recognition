# src/app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from collections import Counter

from . import api

app = FastAPI(title="Industrial Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
frontend_dir = os.path.join(project_root, "src", "frontend")

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/data", StaticFiles(directory=os.path.join(project_root, "data")), name="data")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(frontend_dir, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/datasets")
async def get_datasets():
    processed_dir = os.path.join(project_root, "data", "processed")
    try:
        if os.path.exists(processed_dir):
            datasets = [name for name in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, name))]
            return JSONResponse(content={"datasets": datasets})
        else:
            return JSONResponse(content={"datasets": []})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/example-images/{dataset_name}")
async def get_example_images(dataset_name: str):
    examples = []
    test_dir = os.path.join(project_root, "data", "processed", dataset_name, "test")
    if not os.path.exists(test_dir):
        return JSONResponse(content={"error": "Test directory not found"}, status_code=404)

    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if os.path.isdir(defect_dir):
            for img_file in sorted(os.listdir(defect_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    examples.append({
                        "path": f"data/processed/{dataset_name}/test/{defect_type}/{img_file}",
                        "className": defect_type.replace('_', ' ').title()
                    })
                    break 
    
    return JSONResponse(content={"images": examples})

@app.get("/prediction-log/{dataset_name}")
async def get_prediction_log(dataset_name: str):
    """
    Reads and summarizes the prediction log for a specific dataset.
    """
    log_path = os.path.join(project_root, "prediction_log.jsonl")
    if not os.path.exists(log_path):
        return JSONResponse(content={"summary": {"labels": [], "data": []}, "history": []})

    try:
        defect_counts = Counter()
        history = []
        with open(log_path, 'r') as f:
            for line in f:
                log_entry = json.loads(line)
                if log_entry.get("dataset_name") == dataset_name:
                    history.append(log_entry)
                    defects = log_entry.get("defects_found", {})
                    if not defects:
                        defect_counts["no_defect"] += 1
                    else:
                        for defect_name in defects.keys():
                            defect_counts[defect_name] += 1
        
        # Prepare data for Chart.js
        labels = list(defect_counts.keys())
        data = list(defect_counts.values())
        
        # Return both the summary for the chart and the detailed history
        return JSONResponse(content={"summary": {"labels": labels, "data": data}, "history": history[-20:]}) # Return last 20 entries
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
