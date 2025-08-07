# 🧠 Industrial Defect Detection using U-Net

This project provides an end-to-end solution for industrial defect detection using a **U-Net segmentation model**. It is served via a **FastAPI** backend, packaged and deployed using **Docker**.

---

## 🚀 Getting Started

Follow these steps to prepare the dataset, build the Docker image, and run the API.

---

## 1. 📁 Dataset Setup

- **Dataset**: [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
- **Category used**: `bottle`

### 📂 Required Folder Structure

The application expects the following folder structure inside the `data/processed/` directory:

```
data/
└── processed/
    └── bottle/
        ├── train/
        │   └── good/
        ├── test/
        │   ├── good/
        │   ├── contamination/
        └── ground_truth/
            └── contamination/
```

Each subfolder should contain `.png` images or masks, depending on the type.

---

## 2. 🐳 Docker Deployment

### 🔨 Build the Docker Image

From the root project directory:

```bash
docker build -t defect-detector-app -f deployment/Dockerfile .
```
### 🚀 Run the Docker Container

```bash
docker run -p 8000:8000 --name defect-detector-instance defect-detector-app
```
Or run it using **Docker Desktop** by navigating to the **Images** tab, selecting `defect-detector-app`, clicking **Run**, and publishing port `8000`.

Once running, the API is accessible at:  
👉 `http://127.0.0.1:8000`

---

## ⚙️ API Endpoints

### 🔁 `/train`

- **Purpose**: Trains the U-Net model using the MVTec bottle dataset.
- **Method**: `POST`  
- **URL**: `http://127.0.0.1:8000/train`

---

### 🔍 `/predict`

- **Purpose**: Detects defects in an uploaded image using the trained model.
- **Method**: `POST`  
- **URL**: `http://127.0.0.1:8000/predict`  
- **Body**: `multipart/form-data` with a key `file` (the image to analyze)

---

## ⚠️ Training Notice

You must have a trained a model before using `/predict`. You have two options:

### ✅ Option 1: Train via API (Recommended)

- After starting the container, call the `/train` endpoint.
- The model will be trained inside the container.

### 🖥️ Option 2: Train Locally (Alternative)

- Before building the Docker image, run:

```bash
python src/train.py
```

- This will generate `models/final/best_segmentation_model.pth`.
- When you build the Docker image afterward, it will include this model so you can immediately use `/predict`.

---

## 🧪 Sample Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "path_to_file"
```

---