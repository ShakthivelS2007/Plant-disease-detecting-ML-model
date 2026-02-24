import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# --- THE DEFINITIVE VERSION-COMPATIBILITY PATCH ---
# This looks in all possible hiding places for the Keras layer-loader
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers import serialization

try:
    # Most common path for TF 2.15
    original_get_layer = serialization.get_layer_obj
    target_module = serialization
    target_attr = 'get_layer_obj'
except AttributeError:
    # Fallback for older/newer sub-versions
    original_get_layer = getattr(serialization, 'get_layer', None)
    target_module = serialization
    target_attr = 'get_layer'

def patched_get_layer(config):
    if isinstance(config, dict):
        # Strip Keras 3 metadata that Keras 2 hates
        if 'dtype' in config and isinstance(config['dtype'], dict):
            if 'config' in config['dtype'] and 'name' in config['dtype']['config']:
                config['dtype'] = config['dtype']['config']['name']
        
        config.pop('batch_shape', None)
        config.pop('optional', None)
        config.pop('registered_name', None)
    
    return original_get_layer(config)

# Apply the patch
if original_get_layer:
    setattr(target_module, target_attr, patched_get_layer)

# 1. IMPORT HEATMAP LOGIC
try:
    from test_heatmap import heatmap
except ImportError:
    heatmap = None

app = FastAPI()

# 2. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. MODEL LOADING
MODEL_PATH = "legacy_model.h5"

print("Starting server with Version-Compatibility Patch...")
try:
    # compile=False avoids needing to patch optimizers too
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Legacy model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# 4. API LOGIC
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "msg": "Legacy model deployment successful" if model else "Model failed to load - check logs"
    }

@app.head("/")
async def health_check_head():
    return JSONResponse(content={"status": "online"})

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
        heatmap_file = None
        if heatmap:
            try:
                heatmap_file = heatmap(img_path, model)
            except Exception as he:
                print(f"Heatmap error: {he}")

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
