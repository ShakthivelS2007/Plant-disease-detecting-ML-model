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

# --- THE FIX: Pathing for leaf_model_v1 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this filename EXACTLY matches what you uploaded to GitHub
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

@app.get("/")
async def health():
    """Diagnostic check to see if the file is valid on Render's drive."""
    exists = os.path.exists(MODEL_PATH)
    size = os.path.getsize(MODEL_PATH) if exists else 0
    header_hex = "None"
    if exists:
        with open(MODEL_PATH, 'rb') as f:
            header_hex = f.read(8).hex()
            
    return {
        "status": "online",
        "model_file": "leaf_model_v1.tflite",
        "model_found": exists,
        "size_bytes": size,
        "header_hex": header_hex, # Valid TFLite starts with 0000001c54464c33
        "engine": "TFLite Runtime"
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

        # 2. Check model
        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": f"File {MODEL_PATH} not found"})

        # 3. Load Interpreter
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_idx = interpreter.get_input_details()[0]['index']
        output_idx = interpreter.get_output_details()[0]['index']
        
        # 4. Run Inference
        interpreter.set_tensor(input_idx, img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_idx)[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        heatmap = generate_heatmap_base64(img_array)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": heatmap
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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
