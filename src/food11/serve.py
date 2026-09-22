"""
FastAPI server for the food11 model.

Loads the "champion"-aliased version of the registered `food11` model from
the MLflow Model Registry (not a raw .pth file) and exposes it over HTTP.

Usage:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000

Environment variables:
    MLFLOW_TRACKING_URI - where the MLflow tracking/registry server lives.
                           Defaults to http://127.0.0.1:5000 so it can be
                           overridden from inside a container (e.g. to point
                           at a server running on the host).
"""

import io
import os

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

# --------------------------------------------------------------------------
# Class names, in the same order ImageFolder assigned them during training
# (ImageFolder sorts subfolder names alphabetically). Adjust this list if
# your dataset's class folders are named differently.
# --------------------------------------------------------------------------
CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

MODEL_URI = "models:/food11@champion"

# Same preprocessing used for validation/test in train.py, so inference
# matches what the model was evaluated on.
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = FastAPI(title="food11-serving")

# Populated on startup
model = None


@app.on_event("startup")
def load_model():
    global model
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    input_tensor = preprocess(image).unsqueeze(0)  # add batch dimension
    input_array = input_tensor.numpy().astype(np.float32)

    # mlflow.pyfunc wraps the logged pytorch model; predict() runs a forward
    # pass and returns the raw logits as a numpy array of shape (1, num_classes)
    logits = model.predict(input_array)
    logits = np.asarray(logits)

    # Softmax to turn logits into a confidence score
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    predicted_index = int(np.argmax(probabilities, axis=-1)[0])
    confidence = float(probabilities[0][predicted_index])

    if predicted_index >= len(CLASS_NAMES):
        predicted_label = str(predicted_index)
    else:
        predicted_label = CLASS_NAMES[predicted_index]

    return {
        "category": predicted_label,
        "confidence": confidence,
    }
