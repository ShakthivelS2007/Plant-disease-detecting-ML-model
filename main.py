import os
# Must be at the VERY top
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tf_keras as keras
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
import uuid

# Import your heatmap function
from test_heatmap import heatmap

app = FastAPI()

# Folder setup
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# Load model with a "Safety Net"
MODEL_PATH = "tomato_model_v3_field_ready.h5"
try:
    # compile=False is the strongest way to bypass keyword errors
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Load failed: {e}")
    model = None

@app.get("/")
async def root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        
        # Heatmap
        heatmap_file = heatmap(img_path, model)
        base_url = str(request.base_url).rstrip('/')

        return {
            "prediction": str(idx), 
            "confidence": float(np.max(predictions[0])),
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
