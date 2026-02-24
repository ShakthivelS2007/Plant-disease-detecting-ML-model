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

MODEL_PATH = "legacy_model.h5"
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']
model = None # Initialize globally to avoid "not defined" error

def build_specific_skeleton():
    # Base model without the naming conflict
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    # We build the sequence to match your 3+ layers
    # Note: Keras often gives the base model a default name like 'mobilenetv2_1.00_224'
    m = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
        keras.layers.Dense(128, activation='relu', name="dense"),
        keras.layers.Dropout(0.5, name="dropout"),
        keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name="dense_1")
    ])
    return m

print("🚀 Attempting Final Skeleton Load...")
try:
    model = build_specific_skeleton()
    try:
        # Loading by name to match the keys found in your H5
        model.load_weights(MODEL_PATH, by_name=True)
        print("✅ SUCCESS: Weights loaded into the custom architecture!")
    except Exception as load_err:
        print(f"⚠️ Precise load failed, trying standard weight load: {load_err}")
        model.load_weights(MODEL_PATH)
        print("✅ SUCCESS: Standard weight load worked!")
except Exception as e:
    print(f"❌ FATAL ERROR DURING STARTUP: {e}")
    # We keep 'model' as None so the API doesn't crash, just returns an error

# --- API LOGIC ---
@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

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
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        label = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"prediction": label, "confidence": f"{confidence * 100:.2f}%"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
