# src/app/main.py
from fastapi import FastAPI
from .api import router
# The controller is no longer needed for the startup event
# from .controller import get_inference_engine 

# Initialize the FastAPI app
app = FastAPI(title="Industrial Defect Detection API")

@app.on_event("startup")
def startup_event():
    """
    Application startup event. The inference engine will be loaded on the first
    prediction request instead of here.
    """
    print("--- Application Startup ---")
    # The get_inference_engine() call is removed to fix the error.
    print("--- API is ready. Inference engines will be loaded on demand. ---")

# Include the router from api.py
# All endpoints defined in api.py will now be part of the main app.
app.include_router(router)
