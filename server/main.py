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
    High-Gain Saliency Engine: 
    Forces 'heat' detection by automatically finding the highest contrast 
    areas (Otsu) and amplifying the signal for mobile displays.
    """
    try:
        # 1. Prep raw image (224x224)
        raw_img = (img_array[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Texture Detection (Scharr) - Captures Leaf Curl wrinkles
        # Scharr is more sensitive than Sobel for fine-grain deformities.
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        texture = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, 
                                   cv2.convertScaleAbs(grad_y), 0.5, 0)
        
        # 3. Automatic Necrosis Detection (Otsu) - Captures Blight spots
        # This ignores absolute brightness and finds the darkest 'clusters'.
        _, necrotic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Combine and Amplify (The "Anti-Blue" Fix)
        # We blend texture and spots, then double the intensity.
        combined = cv2.addWeighted(texture, 0.6, necrotic, 0.4, 0)
        combined = cv2.multiply(combined, 2.0) 
        
        # 5. Focus Mask
        # Vignette effect to kill background noise at the edges.
        mask_circle = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(mask_circle, (112, 112), 108, 255, -1)
        combined = cv2.bitwise_and(combined, combined, mask=mask_circle)
        
        # 6. Smooth Glow
        glow = cv2.GaussianBlur(combined, (91, 91), 0)
        heatmap = cv2.applyColorMap(glow, cv2.COLORMAP_JET)
        
        # 7. Masked Overlay
        # Keeps the leaf bright and crisp while painting the red heat.
        mask = glow / 255.0
        mask = np.stack([mask]*3, axis=-1)
        result_img = (raw_img * (1 - mask * 0.4) + heatmap * (mask * 0.9)).astype(np.uint8)
        
        _, buffer = cv2.imencode('.jpg', result_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "online", "viz": "High-Gain_Otsu_v5"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    interpreter = None
    try:
        # Preprocessing
        content = await file.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        if not os.path.exists(MODEL_PATH):
            return JSONResponse(status_code=404, content={"error": "Model missing"})

        # Inference
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        # Visualization
        heatmap_data = generate_heatmap_base64(img_array)

        REMEDIES = [
            [
                "Remove Infected Leaves",
                "Use Fungicides like Chlorothalonil, Mancozeb, Azoxystrobin",
                "Improve air circulation between plants",
            ], 
            
            [
                "Maintain proper watering schedule.",
                "Ensure adequate sunlight exposure.",
                "Monitor for early pest or disease signs.",
            ],
            
            [
                "Control whiteflies using neem oil or approved insecticides.",
                "Remove and destroy infected plants to prevent spread.",
                "Maintain proper plant nutrition and field hygiene."
                
            ],
        ]

        WIKI_URL = [
            "https://en.wikipedia.org/wiki/Alternaria_solani", 
            "None", 
            "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus", 
        ]

        return {
            "prediction": CLASS_NAMES[idx],
            "remedies": REMEDIES[idx],
            "wikipage": WIKI_URL[idx],
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

