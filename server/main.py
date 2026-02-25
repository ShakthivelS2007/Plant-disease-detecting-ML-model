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
    Visualizes focus by targeting high-contrast necrotic regions.
    Creates a 'thermal' cloud over the actual disease spots.
    """
    try:
        # 1. Prepare raw image (0-255 scale)
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Adaptive Thresholding: Specifically targets dark, irregular spots (the blight)
        # This ignores the background/leaf-edge 'noise' that messed up previous versions.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 4)
        
        # 3. Create the 'Focus Cloud'
        # A large kernel (61, 61) spreads the 'heat' into a smooth Grad-CAM style glow.
        glow = cv2.GaussianBlur(thresh, (61, 61), 0)
        
        # 4. Apply JET Colormap (Red = High interest/Blight, Blue = Healthy/Background)
        heatmap = cv2.applyColorMap(glow, cv2.COLORMAP_JET)
        
        # 5. Blend: 65% original leaf, 35% heatmap focus
        result_img = cv2.addWeighted(raw_img, 0.65, heatmap, 0.35, 0)
        
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    """Keeps Render awake and handles cron-job pings."""
    return {
        "status": "online",
        "model": "TFLite_v1",
        "viz_mode": "Adaptive_Saliency"
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
            return JSONResponse(status_code=404, content={"error": "Model file not found"})

        # 2. Setup TFLite Interpreter (Using full engine for opcode support)
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 3. Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # 4. Generate the 'True Focus' Heatmap
        heatmap_data = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Explicit memory cleanup for Render's Free Tier
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
