import os
import uuid
import numpy as np

# 1. SET ENVIRONMENT VARIABLES FIRST
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tf_keras as keras
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# Import your heatmap function (ensure test_heatmap.py is in your repo)
from test_heatmap import heatmap

app = FastAPI()

# 2. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. THE "BRUTE FORCE" MODEL LOADER
MODEL_PATH = "tomato_model_v3_field_ready.h5"

# This mapping forces the loader to ignore the 'batch_shape' error
custom_objs = {
    'InputLayer': tf.keras.layers.Layer
}

print("Attempting to load model...")
try:
    # This is the specific bypass for the InputLayer TypeError
    model = keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects=custom_objs
    )
    print("✅ SUCCESS: Model loaded via Custom Object bypass!")
except Exception as e:
    print(f"⚠️ Method 1 failed: {e}. Trying Method 2 (H5Py)...")
    try:
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            model = keras.models.load_model(f, compile=False)
        print("✅ SUCCESS: Model loaded via H5Py handle!")
    except Exception as e2:
        print(f"❌ FATAL: All loading methods failed: {e2}")
        model = None

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

REMEDIES = {
    'Early Blight': 'Remove infected leaves, use copper-based fungicides, and avoid overhead watering.',
    'Healthy': 'Your plant looks great! Keep maintaining consistent watering and sunlight.',
    'Leaf Curl': 'Check for aphids, use neem oil, or plant resistant varieties.'
}

WIKI_PAGES = {
    'Early Blight': 'https://en.wikipedia.org/wiki/Alternaria_solani',
    'Healthy': 'https://en.wikipedia.org/wiki/Solanum_lycopersicum',
    'Leaf Curl': 'https://en.wikipedia.org/wiki/Leaf_prochlorperazine'
}

@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "engine": "tf-keras-legacy"
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model failed to load on server."})
    
    try:
        # Save uploaded file
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Image Preprocessing
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(np.max(predictions[0]))

        # Heatmap Generation
        heatmap_file = heatmap(img_path, model)

        # Dynamic URLs
        base_url = str(request.base_url).rstrip('/')
        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "remedy": REMEDIES.get(label, "N/A"),
            "wiki_url": WIKI_PAGES.get(label, "#"),
            "original_image_url": f"{base_url}/static/{unique_name}",
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
