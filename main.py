import os
# CRITICAL: This MUST be set before any other imports. 
# It tells TensorFlow to use the Keras 2 saving/loading logic.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

app = FastAPI()

# --- FOLDER SETUP ---
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mount the folder so images are accessible via URL
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# --- MODEL LOADING ---
MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Official Legacy Bridge...")
try:
    # With TF_USE_LEGACY_KERAS=1, this uses the old reliable h5 loader
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# --- API LOGIC ---
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "mode": "Legacy Bridge"
    }

@app.head("/")
async def health_check_head():
    return JSONResponse(content={"status": "online"})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
    try:
        # Save the uploaded file
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Preprocessing
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(np.max(predictions[0]))

        base_url = str(request.base_url).rstrip('/')
        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "image_url": f"{base_url}/static/{unique_name}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
