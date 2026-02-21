import numpy as np
import tensorflow as tf


MODEL_PATH = 'tomato_model_v3_field_ready.h5'

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']


model = tf.keras.models.load_model(MODEL_PATH)

def predict_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
 
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]) 
    
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    return CLASS_NAMES[class_idx], confidence


test_image = r'C:\Users\SAKTHIVEL.S\plant-hack\test_images\leaf1.jpg'
label, conf = predict_disease(test_image)

print("-" * 30)
print(f"RESULT: {label}")
print(f"CONFIDENCE: {conf:.2f}%")
print("-" * 30)
