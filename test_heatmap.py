import numpy as np
import tensorflow as tf
import cv2
import os
import uuid

def heatmap(image_path, model):
    try:
        # 1. Load and Preprocess
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # 2. Find the target layer automatically
        # We look for the last layer with 4D output (batch, h, w, filters)
        target_layer = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                target_layer = layer
                break
        
        if not target_layer:
            print("Could not find a convolutional layer for heatmap.")
            return None

        # 3. Build Grad-Model
        # This maps the input to both the conv layer output and the final prediction
        grad_model = tf.keras.Model(
            [model.inputs], [target_layer.output, model.output]
        )

        # 4. Calculate Gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Get gradients of the top predicted class w.r.t. the output feature map
        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            return None

        # 5. Process Heatmap
        # Mean intensity of the gradients over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        # Multiply each channel by "how important it is" for the prediction
        heatmap_data = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap_data = tf.squeeze(heatmap_data)
        
        # ReLU-like normalization
        heatmap_data = tf.maximum(heatmap_data, 0) / (tf.math.reduce_max(heatmap_data) + 1e-10)
        heatmap_data = heatmap_data.numpy()

        # 6. Overlay onto Original Image
        original_img = cv2.imread(image_path)
        if original_img is None:
            return None
            
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap_data, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply Jet Color Map (Blue = cold, Red = hot)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Superimpose: 60% original image, 40% heatmap
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        # 7. Save to uploads folder
        heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
        save_path = os.path.join("uploads", heatmap_filename)
        cv2.imwrite(save_path, superimposed_img)

        return heatmap_filename

    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        return None
