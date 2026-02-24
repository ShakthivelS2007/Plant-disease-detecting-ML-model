import os
import uuid
import numpy as np
import tensorflow as tf
import tf_keras as keras # Using the legacy bridge

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

app = FastAPI()

# --- FOLDER SETUP ---
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# --- THE BYPASS: BUILDING THE SKELETON ---
def build_skeleton():
    """
    We manually build the architecture so Keras doesn't have to 
    read the corrupted metadata from the .h5 file.
    """
    # 1. Start with the base (This matches most potato/leaf disease models)
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    # 2. Add your specific top layers (GlobalAverage + Dense 3)
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation='softmax') 
    ])
    return model

MODEL_PATH = "legacy_model.h5"

print("🚀 Attempting Weights-Only Load (Bypassing Metadata)...")
try:
    model = build_skeleton()
    # load_weights ONLY looks at the numbers, avoiding the 'as_list' error
    model.load_weights(MODEL_PATH)
    print("✅ SUCCESS: Weights loaded into skeleton!")
except Exception as e:
    print(f"❌ WEIGHT LOAD ERROR: {e}")
    # Fallback: Try a standard load if the skeleton didn't match
    try:
        print("🔄 Attempting standard load as fallback...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ SUCCESS: Standard load worked!")
    except Exception as e2:
        print(f"❌ FATAL ERROR: {e2}")
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

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "image_url": f"{str(request.base_url).rstrip('/')}/static/{unique_name}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    
