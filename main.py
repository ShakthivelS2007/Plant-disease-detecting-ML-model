import os
# MUST be the first thing in the file
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
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# --- MODEL LOADING ---
MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Official Legacy Bridge...")
try:
    # With the env var set, this uses the old h5 engine automatically
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# --- API LOGIC ---
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})
    
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
        label = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "image_url": f"{str(request.base_url).rstrip('/')}/static/{unique_name}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
