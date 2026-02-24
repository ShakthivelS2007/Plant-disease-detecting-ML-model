import os
import uuid
import numpy as np
# Import the legacy engine specifically
import tf_keras as keras
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

# --- THE FIX: CUSTOM INPUT LAYER ---
# This stops the "Unrecognized keyword arguments" crash
class SafeInputLayer(keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Safe Legacy Loader...")
try:
    # Use custom_object_scope to swap the broken InputLayer with our Safe version
    with keras.utils.custom_object_scope({'InputLayer': SafeInputLayer}):
        model = keras.models.load_model(MODEL_PATH, compile=False)
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
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
