import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# Import your heatmap function from test_heatmap.py
from test_heatmap import heatmap

# 1. Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode for Render

app = FastAPI()

# 2. Static Files Setup (for serving heatmaps)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 3. Model Loading 
# compile=False is the key to fixing the "InputLayer" / "batch_shape" error
MODEL_PATH = "tomato_model_v3_field_ready.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

REMEDIES = {
    'Early Blight': 'Remove infected leaves, use copper-based fungicides, and avoid overhead watering.',
    'Healthy': 'Your plant looks great! Keep maintaining consistent watering and sunlight.',
    'Leaf Curl': 'Check for aphids, use neem oil, or plant resistant varieties.'
}

WIKI_PAGES = {
    'Early Blight': 'https://en.wikipedia.org/wiki/Alternaria_solani',
    'Healthy': 'https://en.wikipedia.org/wiki/Solanum_lycopersicum',
    'Leaf Curl': 'https://en.wikipedia.org/wiki/Leaf_prochlorperazine'
}

@app.get("/")
async def read_root():
    return {"message": "Tomato Disease Detection API is Live"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded file with unique ID
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        img_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Preprocess Image
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        label = CLASS_NAMES[idx]
        confidence = float(np.max(predictions[0]))

        # Generate Heatmap (returns only filename)
        heatmap_file = heatmap(img_path, model)

        # Build Full URLs
        base_url = str(request.base_url).rstrip('/')
        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "remedy": REMEDIES.get(label, "N/A"),
            "wiki_url": WIKI_PAGES.get(label, "#"),
            "original_image_url": f"{base_url}/static/{unique_name}",
            "heatmap_url": f"{base_url}/static/{heatmap_file}" if heatmap_file else None
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
