import os
import io
import gc
import base64
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

# --- SMART PATH LOGIC ---
# This ensures Render finds the file regardless of the 'Root Directory' setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

def generate_heatmap_base64(img_array):
    """
    Creates a visual heatmap highlighting high-contrast areas.
    """
    try:
        # Convert to 0-255 uint8 image
        raw_img = (img_array[0] * 255).astype(np.uint8)
        
        # Simple Saliency Map using Sobel gradients
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))
        
        # Apply Jet Color Map (Red = Focus areas)
        heatmap = cv2.applyColorMap(abs_grad, cv2.COLORMAP_JET)
        
        # Combine original + heatmap
        result_img = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

@app.get("/")
async def health():
    """Diagnostic check to verify model status on Render."""
    exists = os.path.exists(MODEL_PATH)
    file_info = "Not Found"
    if exists:
        size = os.path.getsize(MODEL_PATH)
        file_info = f"Found ({size} bytes)"
        # Check first 4 bytes for TFL3 signature
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(4)
            file_info += f" | Header: {header}"
            
    return {
        "status": "online",
        "model_file": file_info,
        "path_searched": MODEL_PATH,
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # 1. Process Image
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        # 2. Safety Check
        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "model.tflite not found at " + MODEL_PATH})

        # 3. Load TFLite Model
        # Using the absolute path ensures no 'Identifier' errors from reading wrong files
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 4. Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # 5. Generate Heatmap
        heatmap_string = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap_string
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Inference Error", "details": str(e)})
    finally:
        if interpreter:
            del interpreter
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
