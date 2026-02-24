import os
import uuid
import numpy as np

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

from test_heatmap import heatmap

app = FastAPI()

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# THE FIX
class FakeDTypePolicy:
    def __init__(self, name="float32", **kwargs): self.name = name
    @classmethod
    def from_config(cls, config): return cls(name=config.get('name', 'float32'))
    def get_config(self): return {"name": self.name}

class FixedInputLayer(keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__(name=kwargs.get('name'))
    @classmethod
    def from_config(cls, config): return cls(name=config.get('name'))

MODEL_PATH = "tomato_model_v3_field_ready.h5"
custom_objs = {
    'InputLayer': FixedInputLayer, 'Layer': FixedInputLayer,
    'DTypePolicy': FakeDTypePolicy, 'DType': FakeDTypePolicy
}

try:
    model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs)
    print("✅ FINAL SUCCESS: Model loaded!")
except Exception as e:
    print(f"❌ FATAL: {e}")
    model = None

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None: return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    try:
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(img_path, "wb") as f: f.write(await file.read())
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
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
