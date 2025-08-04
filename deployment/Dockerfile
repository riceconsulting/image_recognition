# Use an official Python runtime as a parent image
# A slim-buster image is chosen for a smaller footprint
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the file that lists the Python dependencies
COPY deployment/requirements.txt .

# Install the Python dependencies listed in requirements.txt
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code and models
# This includes the src, models, and config directories
COPY ./src ./src
COPY ./models ./models
COPY ./config ./config
COPY ./app ./app

# Expose the port that the API will run on. FastAPI typically uses 8000.
EXPOSE 8000

# Define the command to run the application
# This starts the FastAPI backend service using uvicorn server
# The entry point is the 'app' object in 'app/main.py' [cite: 65, 66]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]