import os
# MANDATORY: Must be set before importing tensorflow/keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tf_keras as keras
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import uuid

# Import your heatmap function
from test_heatmap import heatmap

app = FastAPI()

# Create and mount the uploads folder so images are accessible via URL
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# 1. Load the model using the Legacy Keras engine
MODEL_PATH = "tomato_model_v3_field_ready.h5"
model = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

REMEDIES = {
    'Early Blight': 'Remove infected leaves, use copper-based fungicides, and avoid overhead watering.',
    'Healthy': 'Your plant looks great! Keep maintaining consistent watering and sunlight.',
    'Leaf Curl': 'Check for aphids (vectors), use neem oil, or plant resistant varieties next season.'
}

WIKI_PAGES = {
    'Early Blight': 'https://en.wikipedia.org/wiki/Alternaria_solani',
    'Healthy': 'https://en.wikipedia.org/wiki/Solanum_lycopersicum',
    'Leaf Curl': 'https://en.wikipedia.org/wiki/Leaf_prochlorperazine'
}

@app.get("/")
async def read_root():
    return {"message": "Plant Disease Detection API is Running"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # 2. Save the incoming file with a unique name
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_image_path = os.path.join("uploads", unique_filename)
        
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # 3. Standard Prediction Logic
        img = Image.open(temp_image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        label = CLASS_NAMES[predicted_index]

        # 4. Generate Heatmap (Returns just the filename)
        heatmap_filename = heatmap(temp_image_path, model)

        # 5. Build dynamic URLs for the images
        base_url = str(request.base_url).rstrip('/')
        heatmap_url = f"{base_url}/static/{heatmap_filename}" if heatmap_filename else None
        original_url = f"{base_url}/static/{unique_filename}"

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "remedy": REMEDIES.get(label, "No remedy info available."),
            "wiki_url": WIKI_PAGES.get(label, "#"),
            "heatmap_url": heatmap_url,
            "original_image_url": original_url
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

