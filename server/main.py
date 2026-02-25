import os
import io
import gc
import base64
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "leaf_model_v1.tflite")
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

def generate_heatmap_base64(img_array):
    """
    Generates a smooth 'Focus' heatmap similar to Grad-CAM.
    Uses Gaussian blurring to create the thermal 'cloud' effect.
    """
    try:
        # 1. Prepare the raw image (0-255 scale)
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Thresholding to identify areas of interest
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Apply heavy Gaussian Blur to get that smooth 'thermal' look
        # (55, 55) is the kernel size - larger numbers = smoother clouds
        glow = cv2.GaussianBlur(thresh, (55, 55), 0)
        
        # 4. Apply COLORMAP_JET for the Blue-to-Red rainbow spectrum
        heatmap = cv2.applyColorMap(glow, cv2.COLORMAP_JET)
        
        # 5. Blend: 60% original photo, 40% heatmap overlay
        result_img = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
        
        # Encode to Base64 for Flutter
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

# Handles GET and HEAD requests to keep the server awake
@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    exists = os.path.exists(MODEL_PATH)
    return {
        "status": "online",
        "model_found": exists,
        "engine": "TensorFlow CPU 2.15+",
        "heatmap_mode": "Gaussian Focus",
        "message": "Ready for inference"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # 1. Process Input
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model missing"})

        # 2. Setup TFLite (Full TF Engine for V12 Opcode support)
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 3. Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # 4. Generate Heatmap
        heatmap_data = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Prediction Failed", "details": str(e)})
    finally:
        # Prevent Memory Leaks on Render's Free Tier
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
