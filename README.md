# 🧠 Industrial Defect Detection using Segmentation

This project provides an end-to-end solution for **industrial defect detection** using advanced segmentation models.  
It is served via a **FastAPI** backend, packaged and deployed using **Docker**.

---

## 🚀 Getting Started

Follow these steps to prepare the dataset, build the Docker image, and run the API.

---

## 1. 📁 Dataset Setup

This project used data from MVTEC AD Dataset, you can download it on the link below
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

## ✨ Suitable Images & Capture Conditions

For optimal performance, the quality and consistency of the input images are critical. To achieve the best results, your images should adhere to the following guidelines:

- **Lighting:** Use consistent and diffuse lighting. For transparent objects like bottles, backlighting is highly effective.  
- **Resolution:** Capture high-resolution images (e.g., >1 Megapixel) to ensure small defects are visible.  
- **Focus:** The entire object surface should be in sharp and consistent focus.  
- **Positioning:** Keep the object in a fixed and repeatable position and orientation in every shot.  
- **File Format:** Use lossless formats like `.png` or `.bmp` to avoid compression artifacts.  


---

## 2. 🤖 Available Models

This repository includes **two segmentation models**.  
You can choose which one to train and use via the API.

| Model       | Architecture   | Best For | Notes |
|-------------|----------------|----------|-------|
| **U-Net**   | `unet`         | General purpose, good balance of speed and accuracy | A classic and reliable choice for semantic segmentation |
| **DeepLabV3+** | `deeplabv3plus` | High accuracy, especially with complex defect shapes | Uses a ResNet backbone and is more computationally intensive |

> 💡 Other segmentation architectures like **FCN**, **Mask R-CNN**, or **SegNet** could also be integrated into this framework.

---

## 3. 🐳 Docker Deployment

### 🔨 Build the Docker Image

From the root project directory:

```bash
docker build -t defect-detector-app -f deployment/Dockerfile .
```

### 🚀 Run the Docker Container

```bash
docker run -p 8000:8000 --shm-size="2g" --gpus all --name image-recognition defect-detector-app
```

Or run it using **Docker Desktop** by navigating to the **Images** tab, selecting `defect-detector-app`, clicking **Run**, and publishing port `8000`.

Once running, the API is accessible at:  
👉 `http://127.0.0.1:8000`

---

## ⚙️ API Endpoints

The API allows you to **select which model** to use for training and prediction via a query parameter.

---

### 🔁 `/train`

- **Purpose**: Trains the specified segmentation model  
- **Method**: `POST`  
- **URL**: `http://127.0.0.1:8000/train`  
- **Query Parameter**: `model_arch` (either `unet` or `deeplabv3plus`)

---

### 🔍 `/predict`

- **Purpose**: Detects defects in an uploaded image using the specified model  
- **Method**: `POST`  
- **URL**: `http://127.0.0.1:8000/predict`  
- **Query Parameter**: `model_arch` (either `unet` or `deeplabv3plus`)  
- **Body**: `multipart/form-data` with a key `file` containing the image to analyze

---

## 🧪 Sample API Requests

### Train a specific model

```bash
curl -X POST "http://127.0.0.1:8000/train?model_arch=<model_name>"
```
> **Note:** Replace `<model_name>` with either `unet` or `deeplabv3plus`.
---

### Predict with a specific model

```bash
# Predict using the U-Net model
curl -X POST "http://127.0.0.1:8000/predict?model_arch=<model_name>" \
     -F "image_file=@/path/to/your/image.png"
```
> **Note:** Replace `<model_name>` with either `unet` or `deeplabv3plus`.


## ⚠️ Training Notice

You must have a trained model before using `/predict`. You have two options:

### ✅ Option 1: Train via API (Recommended)
- After starting the container, call the `/train` endpoint.  
- The model will be trained inside the container.

### 🖥️ Option 2: Train Locally (Easy for Docker Rebuild)

You can train either the **U-Net** or **DeepLabV3+** model locally **without starting the API server**.  
The process is controlled by the `config/config.yaml` file.

---

#### **Step 1: Choose the Model to Train**

Open the `config/config.yaml` file and find the `active_model_architecture` key.

Set the value to either:

- `"unet"` or `"deeplabv3plus"`  

Example:

```yaml
active_model_architecture: "unet"  # or "deeplabv3plus"
```
#### **Step 2: Run the training script**

- Before building the Docker image, run:
```bash
python src/train.py
```
This will generate `models/final/best_<model_name>_model.pth`  
*(e.g., `best_unet_model.pth` or `best_deeplabv3_model.pth` depending on the model used).*  

When you build the Docker image afterward, it will include this model,  
allowing you to immediately use `/predict`.

