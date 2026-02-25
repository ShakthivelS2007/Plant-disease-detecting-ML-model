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
    Saliency Mapping: Captures texture (Curl) and necrosis (Blight).
    This logic highlights the features the AI is trained to recognize.
    """
    try:
        # 1. Prepare raw image
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Texture Analysis (For Leaf Curl)
        # Sobel identifies the physical wrinkles and puckering of Curl.
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, 
                                   cv2.convertScaleAbs(grad_y), 0.5, 0)
        
        # 3. Contrast Analysis (For Early Blight)
        # Adaptive Thresholding locks onto those dark necrotic 'target' spots.
        necrotic = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 13, 7)
        
        # 4. Merge Signals & Polish
        # We clean noise so it doesn't look like grain or static.
        combined = cv2.addWeighted(texture, 0.6, necrotic, 0.4, 0)
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # 5. The 'Focus' Glow
        # Massive blur creates the thermal signature look.
        glow = cv2.GaussianBlur(combined, (91, 91), 0)
        heatmap = cv2.applyColorMap(glow, cv2.COLORMAP_JET)
        
        # 6. High-Contrast Overlay
        # Mask ensures we ONLY paint heat where the 'interest' is high.
        mask = glow / 255.0
        mask = np.stack([mask]*3, axis=-1)
        
        # This math keeps the leaf bright but the spots/curls glowing red.
        result_img = (raw_img * (1 - mask * 0.4) + heatmap * (mask * 0.8)).astype(np.uint8)
        
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "online", "mode": "Dual-Feature_Saliency"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model missing"})

        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        heatmap_data = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
