import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

app = FastAPI()

# Config from environment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
api_key_env = os.getenv("API_KEY")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
classifier_model = tf.keras.models.load_model("model.h5")

# VGG19 for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# Target disease classes
class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "Normal"]

def preprocess_image(file: UploadFile):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image = image.resize((180, 180))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    return features.reshape(1, -1)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Check API key
    request_api_key = request.headers.get("x-api-key")
    if request_api_key != api_key_env:
        return JSONResponse(status_code=204, content={})

    # Preprocess image and predict
    features = preprocess_image(file)
    pred = classifier_model.predict(features)[0]
    class_index = np.argmax(pred)
    class_name = class_names[class_index]

    return {
        "class": class_name,
        "confidence": float(pred[class_index])
    }
