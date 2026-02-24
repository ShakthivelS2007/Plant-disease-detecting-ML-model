import numpy as np
import tensorflow as tf
import cv2
import os
import uuid

def heatmap(image_path, model):
    # 1. Load and Preprocess
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 2. Get the specific layer from your base model
    base_model = model.layers[0]
    # 'out_relu' is specific to your model (e.g., MobileNetV2)
    last_conv_layer = base_model.get_layer("out_relu")
    
    grad_model = tf.keras.Model(
        [base_model.inputs], [last_conv_layer.output, base_model.output]
    )
    
    # 3. Calculate Gradients
    with tf.GradientTape() as tape:
        conv_outputs, base_preds = grad_model(img_array)
        
        # Chain the prediction through the rest of the top layers
        x = base_preds
        for layer in model.layers[1:]:
            x = layer(x)
        
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Handle Heatmap Generation
    if grads is None:
        # Fallback if gradients fail
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap_data = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap_data = tf.squeeze(heatmap_data)
    heatmap_data = tf.maximum(heatmap_data, 0) / (tf.math.reduce_max(heatmap_data) + 1e-10)
    heatmap_data = heatmap_data.numpy()

    # 5. Overlay onto the Original Image
    # Load original image for overlay using OpenCV
    original_img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap_data, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # 6. Save with a unique name
    heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
    save_path = os.path.join("uploads", heatmap_filename)
    cv2.imwrite(save_path, superimposed_img)

    return heatmap_filename

