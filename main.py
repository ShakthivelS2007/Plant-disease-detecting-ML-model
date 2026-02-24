import os
import uuid
import json
import numpy as np

# --- THE MONKEYPATCH: This MUST happen before importing Keras ---
import tf_keras.backend as K
from tf_keras.layers import InputLayer

# We are literally rewriting the internal function that is crashing
def fixed_get_input_shape(self):
    if hasattr(self, '_batch_input_shape'):
        shape = self._batch_input_shape
        if isinstance(shape, str):
            try:
                return json.loads(shape)
            except:
                return [None, 224, 224, 3]
        return shape
    return None

# Injecting the fix into the library itself
InputLayer.get_input_shape = fixed_get_input_shape
# -------------------------------------------------------------

import tf_keras as keras
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

app = FastAPI()

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

MODEL_PATH = "legacy_model.h5"

print("🚀 Starting server with Global Monkeypatch...")
try:
    # Now that we've rewritten the internal code, it should load normally
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Model loaded!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    model = None

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
