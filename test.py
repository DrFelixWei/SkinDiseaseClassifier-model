import requests

image_path = r".\data\dermnet\test\Acne and Rosacea Photos\07PerioralDermEye.jpg"
api_key = "your_api_key_here"

with open(image_path, "rb") as img:
    files = {"file": img}
    headers = {"x-api-key": api_key}
    response = requests.post("http://localhost:8000/predict", files=files, headers=headers)
    print("Prediction:", response.json())
