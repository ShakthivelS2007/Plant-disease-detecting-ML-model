import os
import uuid
import numpy as np
import tensorflow as tf
import tf_keras as keras
import h5py
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

MODEL_PATH = "legacy_model.h5"
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

# --- THE INSPECTOR ---
def get_h5_layers(path):
    print("🔍 [STEP 1] INSPECTING H5 FILE...")
    try:
        with h5py.File(path, 'r') as f:
            # Check for weights keys
            if 'model_weights' in f:
                layers = list(f['model_weights'].keys())
                print(f"📦 [INFO] Layers found in file: {layers}")
                return layers
            else:
                print("⚠️ [WARN] 'model_weights' key not found in H5.")
    except Exception as e:
        print(f"❌ [ERROR] Inspection failed: {e}")
    return []

# --- THE SKELETON BUILDER ---
def build_3_layer_skeleton():
    # This architecture specifically creates 3 distinct top-level layers
    # to match the "found 3 saved layers" error.
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    # We wrap the base model to ensure it counts as exactly ONE layer
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3), name="input_layer"), # Layer 1
        base_model,                                                            # Layer 2
        keras.layers.Sequential([                                              # Layer 3
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ], name="top_head")
    ])
    return model

print("🚀 Starting Server...")
detected_layers = get_h5_layers(MODEL_PATH)

try:
    print("🛠️ [STEP 2] Building 3-layer skeleton...")
    model = build_3_layer_skeleton()
    print("📥 [STEP 3] Attempting weight load...")
    model.load_weights(MODEL_PATH)
    print("✅ SUCCESS: Weights matched and loaded!")
except Exception as e:
    print(f"❌ SKELETON LOAD FAILED: {e}")
    model = None

# --- API ROUTES ---
@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "h5_layers": detected_layers
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not initialized."})
    
    try:
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
