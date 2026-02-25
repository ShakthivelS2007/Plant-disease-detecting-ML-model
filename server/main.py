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
    """Generates a visual heatmap for the Flutter UI."""
    try:
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))
        heatmap = cv2.applyColorMap(abs_grad, cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

# Updated this route to accept HEAD requests for the Cron-job pinger
@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    """Diagnostic check for model health and pinger compatibility."""
    exists = os.path.exists(MODEL_PATH)
    size = os.path.getsize(MODEL_PATH) if exists else 0
    return {
        "status": "online",
        "model_file": "leaf_model_v1.tflite",
        "model_found": exists,
        "engine": "TensorFlow CPU (Full)",
        "message": "Server is awake and ready!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # 1. Read and Preprocess Image
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        # 2. Check for Model
        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model file not found"})

        # 3. Load TFLite Interpreter via TensorFlow
        # Using tf.lite for maximum opcode version support
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 4. Run Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # 5. Get Heatmap Data
        heatmap_data = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Inference Error", "details": str(e)})
    finally:
        # Crucial for Render's 512MB RAM limit
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
