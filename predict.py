import numpy as np
import tensorflow as tf
import os

# --- UPDATED TO USE YOUR NEW LEGACY MODEL ---
MODEL_PATH = 'legacy_model.h5' 
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

# Check if file exists before loading to avoid ugly crashes
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found! Make sure it's in the same folder.")
    exit()

# Load with compile=False to avoid the 'metrics' warnings
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_disease(image_path):
    if not os.path.exists(image_path):
        return "Image not found", 0.0

    # Load and resize
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    # Preprocess (Convert to array and normalize to 0-1)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Run prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Find the highest score
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    return CLASS_NAMES[class_idx], confidence

# --- TEST IT OUT ---
test_image = r'C:\Users\SAKTHIVEL.S\plant-hack\test_images\leaf1.jpg'

print(f"Analyzing: {test_image}...")
label, conf = predict_disease(test_image)

print("-" * 30)
print(f"RESULT    : {label}")
print(f"CONFIDENCE: {conf:.2f}%")
print("-" * 30)
