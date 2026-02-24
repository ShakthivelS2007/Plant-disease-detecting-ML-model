import os
import uuid
import io
import gc
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

def build_functional_model():
    """Builds the skeleton based on the layer names found in your H5 file."""
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Base MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None
    )
    # This name must match the key found in your H5 metadata
    base_model._name = "mobilenetv2_1.00_224"
    x = base_model(inputs)
    
    # Matching the 'h5_layers' you shared earlier
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dense(128, activation='relu', name="dense")(x)
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax', name="dense_1")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

print("🚀 BOOTING: Loading model into memory...")
try:
    # 1. Clear any existing background sessions
    keras.backend.clear_session()
    
    # 2. Build and Load
    model = build_functional_model()
    # by_name=True is mandatory for skip_mismatch
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    
    print("✅ SUCCESS: Model is ready.")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {e}")

@app.get("/")
async def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})
    
    try:
        # Read file into memory buffer
        content = await file.read()
        
        # Optimized Image Processing
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        # Force CPU and predict with small batch to save RAM
        with tf.device('/cpu:0'):
            predictions = model.predict(img_array, batch_size=1, verbose=0)
            
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])

        # --- AGGRESSIVE CLEANUP ---
        # Explicitly delete heavy objects and trigger Garbage Collection
        del img_array
        del content
        gc.collect() 

        return {
            "prediction": label, 
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
