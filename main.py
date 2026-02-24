import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# --- NO MORE PATCHES NEEDED WITH TF 2.15 ---

app = FastAPI()

# 1. FOLDER SETUP
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure the static files are mounted so you can view images/heatmaps in browser
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 2. MODEL LOADING
# We use .h5 format. With TF 2.15, this should load without the 'as_list' error.
MODEL_PATH = "legacy_model.h5"

print("Loading model using TensorFlow 2.15 compatibility mode...")
try:
    # compile=False is the key to avoiding optimizer/version errors
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Legacy model loaded perfectly!")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# 3. CLASS NAMES
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

# 4. API ROUTES
@app.get("/")
async def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "message": "Plant Disease Detection API is running"
    }

@app.head("/")
async def health_check_head():
    # Render uses HEAD requests to check if your app is alive
    return JSONResponse(content={"status": "online"})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})
    
    try:
        # Save uploaded file
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Preprocessing (MobileNet/Standard CNN standard)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(np.max(predictions[0]))

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "image_url": f"{str(request.base_url).rstrip('/')}/static/{unique_name}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
