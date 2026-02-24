import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# --- THE "GOD MODE" CUSTOM OBJECT SCOPE ---
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, BatchNormalization, MaxPooling2D, DepthwiseConv2D, Flatten, Dropout

def strip_keras3(kwargs):
    """Utility to clean up Keras 3 metadata for Keras 2"""
    kwargs.pop('batch_shape', None)
    kwargs.pop('optional', None)
    kwargs.pop('registered_name', None)
    if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
        kwargs['dtype'] = kwargs.get('dtype', {}).get('config', {}).get('name', 'float32')
    return kwargs

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs): super().__init__(*args, **strip_keras3(kwargs))

class PatchedConv2D(Conv2D):
    def __init__(self, *args, **kwargs): super().__init__(*args, **strip_keras3(kwargs))

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs): super().__init__(*args, **strip_keras3(kwargs))

class PatchedBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs): super().__init__(*args, **strip_keras3(kwargs))

# 1. IMPORT HEATMAP LOGIC
try:
    from test_heatmap import heatmap
except ImportError:
    heatmap = None

app = FastAPI()

# 2. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. MODEL LOADING
MODEL_PATH = "legacy_model.h5"

print("Starting server with God Mode Custom Scope...") # <--- LOOK FOR THIS!
try:
    custom_objects = {
        'InputLayer': PatchedInputLayer,
        'Conv2D': PatchedConv2D,
        'DepthwiseConv2D': PatchedDepthwiseConv2D,
        'BatchNormalization': PatchedBatchNormalization
    }
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Legacy model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# ... (The rest of your API routes remain the same)
@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.head("/")
async def health_check_head():
    return JSONResponse(content={"status": "online"})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None: return JSONResponse(status_code=500, content={"error": "Model not loaded."})
    try:
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(img_path, "wb") as f: f.write(await file.read())
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        label = ['Early Blight', 'Healthy', 'Leaf Curl'][np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        base_url = str(request.base_url).rstrip('/')
        return {"prediction": label, "confidence": f"{confidence * 100:.2f}%"}
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
