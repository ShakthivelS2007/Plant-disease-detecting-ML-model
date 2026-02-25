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
    Targets the 'Necrotic Core' of the disease. 
    Uses morphological opening to remove noise and a huge Gaussian blur 
    to create a professional-grade diagnostic 'glow'.
    """
    try:
        # 1. Image Prep
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Adaptive Thresholding: Only grab pixels much darker than their neighbors
        # This effectively ignores the bright background and highlights dark spots.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 13, 6)
        
        # 3. Morphology: Remove tiny specks/noise (salt & pepper effect)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 4. Create the 'Focus Cloud': A massive 91x91 blur for that 'AI Brain' look
        glow = cv2.GaussianBlur(thresh, (91, 91), 0)
        
        # 5. Apply JET (Blue-to-Red thermal spectrum)
        heatmap = cv2.applyColorMap(glow, cv2.COLORMAP_JET)
        
        # 6. Blend: 75% original leaf, 25% heatmap focus
        result_img = cv2.addWeighted(raw_img, 0.75, heatmap, 0.25, 0)
        
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "online", "viz": "Morphological_Saliency_v2"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # Process Input
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model missing"})

        # TFLite Inference
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # Generate the 'Clean' Heatmap
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
