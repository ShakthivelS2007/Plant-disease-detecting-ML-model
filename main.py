import os
import uuid
import numpy as np
import tensorflow as tf
import tf_keras as keras

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

# --- THE UNIVERSAL SCRUBBER ---
def clean_config(config):
    """Recursively removes Keras 3 metadata from any layer configuration"""
    if not isinstance(config, dict):
        return config
    
    # Remove Keras 3 specific keys that break Keras 2
    keys_to_pop = ['batch_shape', 'optional', 'registered_name', 'module']
    for key in keys_to_pop:
        config.pop(key, None)
    
    # Fix the DTypePolicy/dtype issue
    if 'dtype' in config and isinstance(config['dtype'], dict):
        # Extract just the string name (e.g., 'float32')
        config['dtype'] = config['dtype'].get('config', {}).get('name', 'float32')
        
    # Recursively clean nested dictionaries (like initializers)
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = clean_config(value)
            
    return config

# Create a 'Global Wrapper' for any layer class
def wrap_layer(layer_cls):
    class SafeLayer(layer_cls):
        @classmethod
        def from_config(cls, config):
            return super(SafeLayer, cls).from_config(clean_config(config))
    return SafeLayer

MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Universal Config Scrubber...")
try:
    # We wrap the most common layers that cause these 'deserialization' errors
    patched_objects = {
        'InputLayer': wrap_layer(keras.layers.InputLayer),
        'Conv2D': wrap_layer(keras.layers.Conv2D),
        'DepthwiseConv2D': wrap_layer(keras.layers.DepthwiseConv2D),
        'BatchNormalization': wrap_layer(keras.layers.BatchNormalization),
        'Dense': wrap_layer(keras.layers.Dense),
        'ReLU': wrap_layer(keras.layers.ReLU)
    }
    
    with keras.utils.custom_object_scope(patched_objects):
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

        return {"prediction": label, "confidence": f"{confidence * 100:.2f}%"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    
