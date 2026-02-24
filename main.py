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

# --- THE FIX: DYNAMIC SKELETON ---
def build_model_v3():
    # This structure covers 99% of plant disease models made in Colab/Kaggle
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)), # Layer 1
        base_model,                                       # Layer 2
        keras.layers.GlobalAveragePooling2D(),            # Layer 3
        keras.layers.Dense(3, activation='softmax')       # Layer 4
    ])
    return model

print("🚀 Attempting Universal Skeleton Load...")
try:
    model = build_model_v3()
    try:
        model.load_weights(MODEL_PATH)
        print("✅ SUCCESS: Weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ Layer mismatch on 4-layer skeleton. Trying 3-layer...")
        # If 4 layers failed, try the 3-layer version
        model = keras.Sequential([
            keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(3, activation='softmax')
        ])
        model.load_weights(MODEL_PATH)
        print("✅ SUCCESS: Weights loaded into 3-layer skeleton!")
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
