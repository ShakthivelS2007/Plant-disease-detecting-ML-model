import os
import io
import gc
import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

# --- CONFIGURATION ---
MODEL_PATH = "model.tflite"
# Ensure these match the order your model was trained on!
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

# --- MODEL INITIALIZATION ---
def get_interpreter():
    """Load the TFLite model and allocate tensors."""
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# --- PREDICTION LOGIC ---
def run_inference(img_array):
    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure input matches model's expected shape/type
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Extract results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# --- API ENDPOINTS ---

@app.get("/")
async def health_check():
    """Confirm the server is up and the model file is accessible."""
    exists = os.path.exists(MODEL_PATH)
    return {
        "status": "online",
        "model_found": exists,
        "engine": "TFLite Runtime",
        "root_directory": "server"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded file
        content = await file.read()
        
        # 2. Preprocess the image
        with Image.open(io.BytesIO(content)) as img:
            # Convert to RGB (removes Alpha channel if it's a PNG)
            img = img.convert("RGB")
            # Resize to the input size your model expects (usually 224x224)
            img = img.resize((224, 224))
            
            # Convert to Numpy and Normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

        # 3. Run the TFLite Inference
        predictions = run_inference(img_array)
        
        # 4. Process Results
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[predicted_index])
        
        response_data = {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": f"{confidence * 100:.2f}%",
            "all_scores": {
                name: f"{float(score) * 100:.2f}%" 
                for name, score in zip(CLASS_NAMES, predictions)
            }
        }

        # 5. Manual Garbage Collection (Crucial for 512MB RAM)
        del img_array
        del content
        gc.collect()

        return response_data

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": "Prediction failed", "details": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
