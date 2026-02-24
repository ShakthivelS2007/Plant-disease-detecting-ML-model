from fastapi import FastAPI, File, UploadFile, Request # Added Request
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

# --- RENDER FIX: Disable GPU to save RAM ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_uploads():
    now = time.time()
    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)
        # Check if file is older than 10 mins
        if os.stat(path).st_mtime < now - 600:
            try:
                os.remove(path)
            except:
                pass

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (Ensure this file is in your GitHub repo root)
model = tf.keras.models.load_model("tomato_model_v3_field_ready.h5")

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
def home():
    return {"message": "Plant Detection API Running"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)): # Added request here
    try:
        clean_uploads()
        contents = await file.read()

        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(image_path, "wb") as f:
            f.write(contents)
        
        heatmap_data = heatmap(image_path, model)

        # Preprocess
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224)) 
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        REMEDIES = [
            ["Remove infected leaves immediately.", "Use copper based fungicide.", "Ensure proper air circulation."],
            ["Continue watering schedule.", "Ensure good sunlight exposure.", "Monitor early signs of pest."],
            ["Control whiteflies using neem oil.", "Remove infected plants.", "Maintain proper plant nutrition."]
        ]

        WIKI_PAGES = [
            "https://en.wikipedia.org/wiki/Alternaria_solani",
            "None",
            "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus",
        ]

        # --- RENDER FIX: Dynamic Heatmap URL ---
        # We replace the local IP (192.168.1.5) with the actual URL of your Render service
        base_url = str(request.base_url).rstrip("/")
        heatmap_url = f"{base_url}/uploads/{heatmap_data}"

        return {
            "success": True,
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2),
            "remedies": REMEDIES[predicted_index],
            "wikipage": WIKI_PAGES[predicted_index],
            "heatmap": heatmap_url,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
