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

# FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# --- THE UNIVERSAL CONFIG SCRUBBER ---
def clean_config(config):
    if not isinstance(config, dict):
        return config
    
    # Kill Keras 3 specific keys
    for key in ['batch_shape', 'optional', 'registered_name', 'module']:
        config.pop(key, None)
    
    # Convert DTypePolicy dict to a simple string
    if 'dtype' in config and isinstance(config['dtype'], dict):
        config['dtype'] = config['dtype'].get('config', {}).get('name', 'float32')
        
    # Deep clean nested structures
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = clean_config(value)
    return config

# --- THE FAULT-TOLERANT LAYER PROXY ---
class UniversalLayerProxy:
    def get_layer(self, name):
        # Try standard layers first, then preprocessing
        layer_cls = getattr(keras.layers, name, None)
        if layer_cls is None:
            layer_cls = getattr(keras.layers.experimental.preprocessing, name, None)
        
        if layer_cls is None or not isinstance(layer_cls, type):
            return None
            
        class SafeLayer(layer_cls):
            @classmethod
            def from_config(cls, config):
                return super(SafeLayer, cls).from_config(clean_config(config))
        return SafeLayer

MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Fault-Tolerant Layer Proxy...")
try:
    # A massive list to catch everything. If it doesn't exist, it's ignored.
    layer_names = [
        'InputLayer', 'Conv2D', 'DepthwiseConv2D', 'BatchNormalization', 
        'ReLU', 'MaxPooling2D', 'GlobalAveragePooling2D', 'Dense', 
        'Dropout', 'Flatten', 'ZeroPadding2D', 'Add', 'Rescale', 'Activation',
        'Resizing', 'Normalization'
    ]
    
    proxy = UniversalLayerProxy()
    custom_objects = {}
    for name in layer_names:
        patched = proxy.get_layer(name)
        if patched:
            custom_objects[name] = patched
    
    with keras.utils.custom_object_scope(custom_objects):
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
