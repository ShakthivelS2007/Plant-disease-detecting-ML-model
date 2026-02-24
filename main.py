import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# --- THE UNIVERSAL REPAIR PATCH ---
from tensorflow.keras import layers

def strip_keras3(kwargs):
    """Safely removes Keras 3 metadata without breaking internal Keras 2 logic"""
    kwargs.pop('batch_shape', None)
    kwargs.pop('optional', None)
    kwargs.pop('registered_name', None)
    
    # If dtype is a Keras 3 dictionary (DTypePolicy), delete it.
    # Keras 2 will then automatically default to 'float32', avoiding the 'str' error.
    if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
        kwargs.pop('dtype') 
    return kwargs

def patch_layer(layer_class):
    class PatchedLayer(layer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **strip_keras3(kwargs))
    return PatchedLayer

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

print("Starting server with Universal Repair Scope...")
try:
    # Comprehensive list of layer types to patch
    layer_types = [
        'InputLayer', 'Conv2D', 'DepthwiseConv2D', 'BatchNormalization', 
        'ReLU', 'MaxPooling2D', 'GlobalAveragePooling2D', 'Dense', 
        'Dropout', 'Flatten', 'ZeroPadding2D', 'Add', 'Rescale', 'Activation'
    ]
    
    custom_objects = {}
    for name in layer_types:
        if hasattr(layers, name):
            custom_objects[name] = patch_layer(getattr(layers, name))
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Legacy model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# 4. API LOGIC
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

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
