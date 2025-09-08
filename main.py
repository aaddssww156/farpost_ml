import os
import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serving")

TEXT_MODEL_PATH = "models/text_model.pkl"
IMAGE_MODEL_PATH = "models/image_model.pth"

# Для использования cuda ядер
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

text_model = joblib.load(TEXT_MODEL_PATH)
logger.info(f"Loaded text model: {TEXT_MODEL_PATH}")

image_model_state = torch.load(IMAGE_MODEL_PATH, map_location=torch.device(DEVICE))

class ImageModel(nn.Module):
    def __init__(self,):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class_names = image_model_state["classes"]
image_model = ImageModel()
image_model.load_state_dict(image_model_state["model_state_dict"])
image_model.to(DEVICE)
image_model.eval()

logger.info(f"Loaded image model: {IMAGE_MODEL_PATH}")

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"} 

@app.post("/v1/predict_text")
def predict_text(payload: dict):
    if "text" not in payload:
        raise HTTPException(status_code=400, detail="Text required")
    logger.info(f"Text prediction, request: {payload['text']}")
    pred = text_model.predict([payload["text"]])[0]

    return {
        "predicted_class": pred,
    } 


@app.post("/v1/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Image could not be processed")


    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = image_model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_idx = torch.argmax(probabilities).item()

    return {
            "predicted_class": class_names[predicted_class_idx],
            "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    }
