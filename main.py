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
from test_heatmap import heatmap

app = FastAPI()

# 2. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. THE AGGRESSIVE INPUTLAYER FIX
# This intercepts the bad config and cleans it before Keras can complain.
class FixedInputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        # Remove the keywords that cause the 'Unrecognized keyword' error
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('batch_shape', None)
        config.pop('optional', None)
        return cls(**config)

MODEL_PATH = "tomato_model_v3_field_ready.h5"
custom_objs = {
    'InputLayer': FixedInputLayer,
    'Layer': FixedInputLayer # Some versions call it Layer instead of InputLayer
}

print("Final attempt at model loading...")
try:
    # We use custom_objects to swap the broken InputLayer with our Fixed version
    model = keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects=custom_objs
    )
    print("✅ SUCCESS: Model loaded and InputLayer bypassed!")
except Exception as e:
    print(f"❌ INTERNAL ERROR: {e}")
    model = None

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "details": "Model failed to load" if model is None else "Ready for prediction"
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
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
        
        heatmap_file = heatmap(img_path, model)
        base_url = str(request.base_url).rstrip('/')
        
        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{float(np.max(predictions[0])) * 100:.2f}%",
            "original_image_url": f"{base_url}/static/{unique_name}",
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
