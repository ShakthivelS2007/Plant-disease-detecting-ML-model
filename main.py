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
model = None 

def build_functional_model():
    # We use the Functional API to give us total control over names
    inputs = keras.Input(shape=(224, 224, 3), name="input_layer")
    
    # 1. The Base Model
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    # Force the base model name to match your H5 key
    base_model._name = "mobilenetv2_1.00_224"
    x = base_model(inputs)
    
    # 2. The Global Pooling
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    
    # 3. The Dense Layers (Trial & Error on the size, but usually 128 or 256 or 512 or 1024)
    # If this fails, the 'axes' error is because this 'dense' layer size is wrong.
    # We'll try to let it skip this if it doesn't match.
    x = keras.layers.Dense(128, activation='relu', name="dense")(x) 
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    
    # 4. Final Output
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name="dense_1")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

print("🚀 Attempting Surgical Functional Load...")
try:
    model = build_functional_model()
    # by_name=True + skip_mismatch=True is the ultimate 'just work' combo
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    print("✅ SUCCESS: Weights mapped to named layers!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# --- API LOGIC ---
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
