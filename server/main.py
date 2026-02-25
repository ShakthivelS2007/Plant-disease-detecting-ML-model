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
    """Generates a visual heatmap. Changed to COLORMAP_HOT for better clarity."""
    try:
        # Convert back to 0-255 scale
        raw_img = (img_array[0] * 255).astype(np.uint8)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # Find gradients (the 'activity' in the image)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))
        
        # APPLY COLORMAP_HOT: Areas of high change glow orange/white, low change stays dark
        heatmap = cv2.applyColorMap(abs_grad, cv2.COLORMAP_HOT)
        
        # Blend the original image with the heatmap
        result_img = cv2.addWeighted(raw_img, 0.7, heatmap, 0.3, 0)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

# Supports GET and HEAD to keep Render awake without 405 errors
@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    """Health check for pinger and basic diagnostics."""
    exists = os.path.exists(MODEL_PATH)
    return {
        "status": "online",
        "model_found": exists,
        "engine": "TensorFlow CPU 2.15+",
        "message": "Ready for inference"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # 1. Load and Resize
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model missing"})

        # 2. Setup TFLite Interpreter
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
        # Help Render's RAM management
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
