import os
import uuid
import numpy as np

# 1. THE ESSENTIAL ENVIRONMENT OVERRIDES
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

# Ensure test_heatmap.py is in your repository!
try:
    from test_heatmap import heatmap
except ImportError:
    heatmap = None
    print("⚠️ Warning: test_heatmap.py not found. Heatmaps will be disabled.")

app = FastAPI()

# 2. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. THE "SCORCHED EARTH" TRANSLATOR
# This class tricks the old Keras into accepting new Keras 3 'DTypePolicy' tags
class FakeDTypePolicy:
    def __init__(self, name="float32", **kwargs): 
        self.name = name
    @classmethod
    def from_config(cls, config): 
        return cls(name=config.get('name', 'float32'))
    def get_config(self): 
        return {"name": self.name}

# This class ignores ALL metadata from the H5 file to prevent keyword errors
class FixedInputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        # We only keep the name, ignoring batch_shape, sparse, ragged, etc.
        super().__init__(name=kwargs.get('name'))

    @classmethod
    def from_config(cls, config):
        # Strip everything except the name
        return cls(name=config.get('name'))

MODEL_PATH = "tomato_model_v3_field_ready.h5"

# Map every possible problematic layer name to our fixed versions
custom_objs = {
    'InputLayer': FixedInputLayer,
    'Layer': FixedInputLayer,
    'DTypePolicy': FakeDTypePolicy,
    'DType': FakeDTypePolicy
}

print("Attempting 'Scorched Earth' model load...")
try:
    # We bypass the 'InputLayer' config entirely using custom_objects
    model = keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects=custom_objs
    )
    print("✅ FINAL SUCCESS: Model loaded and all bad keywords ignored!")
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
        "msg": "If model_loaded is true, the war is won."
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
    try:
        # Save uploaded file
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Image Preprocessing (Matches your training dimensions)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(np.max(predictions[0]))

        # Heatmap Generation
        heatmap_file = None
        if heatmap:
            try:
                heatmap_file = heatmap(img_path, model)
            except Exception as he:
                print(f"Heatmap error: {he}")

        # Dynamic URLs
        base_url = str(request.base_url).rstrip('/')
        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "original_image_url": f"{base_url}/static/{unique_name}",
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
