import requests
import numpy as np
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
import os

# Load VGG19 model for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((180, 180))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # important for VGG19
    features = vgg_model.predict(img_array)
    return features.reshape(1, -1)

# Image path
image_path = r".\data\dermnet\test\Acne and Rosacea Photos\07PerioralDermEye.jpg"
# image_path = r".\data\dermnet\test\Normal\normal_0e19f8a9b2f135ed3acb_jpg.rf.7b0729f4b66cb9504413e06459509e7f.jpg"

# Prepare payload
features = preprocess_image(image_path)
payload = {
    "data": features.tolist()
}

# Send to local API
response = requests.post("http://localhost:8000/predict", json=payload)
print("Prediction:", response.json())
