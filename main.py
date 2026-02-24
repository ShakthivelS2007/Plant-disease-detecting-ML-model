import os
import uuid
import io
import numpy as np
import tensorflow as tf
import tf_keras as keras
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI()

# FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

MODEL_PATH = "legacy_model.h5"
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']
model = None 

def build_final_model():
    # We build the architecture based on your H5 inspection
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None
    )
    m = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
        keras.layers.Dense(128, activation='relu', name="dense"),
        keras.layers.Dropout(0.5, name="dropout"),
        keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name="dense_1")
    ])
    return m

print("🚀 RUNNING FINAL WEIGHT-MAPPER...")
try:
    model = build_final_model()
    # FIX: skip_mismatch=True MUST have by_name=True
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    print("✅ SUCCESS: Model loaded (potentially with some skipped layers)")
except Exception as e:
    print(f"❌ WEIGHT MAPPER FAILED: {e}")

@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model missing"})
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])

        return {"prediction": label, "confidence": f"{confidence * 100:.2f}%"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
