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

# 3. THE LAMBDA "TRANSLATOR" FIX
MODEL_PATH = "tomato_model_v3_field_ready.h5"

# This function manually cleans the 'InputLayer' config as it loads
def fix_input_layer(**kwargs):
    # Rename batch_shape to input_shape and remove 'optional'
    config = kwargs.copy()
    if 'batch_shape' in config:
        # Convert [None, 224, 224, 3] to (224, 224, 3)
        batch_shape = config.pop('batch_shape')
        config['input_shape'] = tuple(batch_shape[1:])
    config.pop('optional', None)
    return keras.layers.InputLayer(**config)

custom_objs = {'InputLayer': fix_input_layer}

print("Attempting final model load...")
try:
    model = keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects=custom_objs
    )
    print("✅ SUCCESS: Model loaded and InputLayer fixed!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# ... rest of your class names and remedies
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "msg": "If model_loaded is true, you can now use the /predict endpoint!"
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
