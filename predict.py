import numpy as np
import tensorflow as tf
import os


MODEL_PATH = 'legacy_model.h5' 
CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']


if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found! Make sure it's in the same folder.")
    exit()


model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_disease(image_path):
    if not os.path.exists(image_path):
        return "Image not found", 0.0

    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    
    predictions = model.predict(img_array, verbose=0)
    
    
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    return CLASS_NAMES[class_idx], confidence


test_image = r'C:\Users\SAKTHIVEL.S\plant-hack\test_images\leaf1.jpg'

print(f"Analyzing: {test_image}...")
label, conf = predict_disease(test_image)

print("-" * 30)
print(f"RESULT    : {label}")
print(f"CONFIDENCE: {conf:.2f}%")
print("-" * 30)

