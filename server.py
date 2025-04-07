import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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

model = tf.keras.models.load_model("model.h5")
class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea"]

@app.post("/predict")
async def predict(request: Request):
    request_api_key = request.headers.get("x-api-key")

    if request_api_key != api_key_env:
        return {}

    # Parse and predict
    data = await request.json()
    input_features = np.array(data["data"])
    pred = model.predict(input_features)[0]
    class_index = np.argmax(pred)
    class_name = class_names[class_index]
    return {
        "class": class_name,
        "confidence": float(pred[class_index])
    }
