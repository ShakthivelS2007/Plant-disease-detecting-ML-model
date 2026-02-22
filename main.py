from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from test_heatmap import heatmap
import os
import uuid
from fastapi.staticfiles import StaticFiles
import time

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_uploads():
    now = time.time()
    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.stat(path).st_mtime < now - 600:
            os.remove(path)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Connection with Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .h5 model
model = tf.keras.models.load_model("tomato_model_v3_field_ready.h5")

# Class names 
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
def home():
    return {"message": "Plant Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        clean_uploads()
        # Read image
        contents = await file.read()

        # Create unique filename
        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save image to disk
        with open(image_path, "wb") as f:
            f.write(contents)
        
        heatmap_data = heatmap(image_path, model)

        #Preprocess
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224)) 
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        REMEDIES = [
            [
                "Remove inf ected leaves immediately.",
                "Use copper based fungicide (eg: chlorothalonil, mancozeb, azoxystrobin, etc).",
                "Ensure there is proper air circulation between plants.",
            ],
            [
                "Continue watering schedule.",
                "Ensure good sunlight exposure.",
                "Mointor early signs of pest.",
            ],
            [
                "Control whitefiles using neem oil.",
                "Remove infected plants to prevent spread.",
                "Maintain proper plant nutrition.",
            ],
        ]

        WIKI_PAGES = [
            [
                "https://en.wikipedia.org/wiki/Alternaria_solani",
            ],
            [
                "-"
            ],
            [
                "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus",
            ],
        ]

        return {
            "success": True,
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2),
            "remedies": REMEDIES[predicted_index],
            "wikipage": WIKI_PAGES[predicted_index],
            "heatmap": f"http://192.168.1.5:8000/uploads/{heatmap_data}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }