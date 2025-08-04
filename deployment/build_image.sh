#!/bin/bash

# This script builds the Docker image for the defect detection application [cite: 87]

# Set the name and tag for the Docker image
IMAGE_NAME="industrial-defect-detection"
IMAGE_TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Run the docker build command
# The '.' specifies that the build context is the current directory, where the Dockerfile is located
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Build complete."