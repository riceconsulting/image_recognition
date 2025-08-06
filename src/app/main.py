# src/app/main.py
from fastapi import FastAPI
from .api import router
from .controller import get_inference_engine

# Initialize the FastAPI app
app = FastAPI(title="Industrial Defect Detection API")

@app.on_event("startup")
def startup_event():
    """
    On startup, this will initialize the inference engine so the first
    prediction is fast.
    """
    print("--- Application Startup ---")
    get_inference_engine()
    print("--- API is ready ---")

# Include the router from api.py
# All endpoints defined in api.py will now be part of the main app.
app.include_router(router)
