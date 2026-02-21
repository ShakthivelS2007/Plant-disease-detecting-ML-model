import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

CLASS_NAMES = ['Early Blight', 'Healthy', 'Leaf Curl']

def heatmap(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)

    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("out_relu")
    
    grad_model = tf.keras.Model(
    [base_model.inputs], [last_conv_layer.output, base_model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, base_preds = grad_model(img_array)
        
        x = base_preds
        
        for layer in model.layers[1:]:
            x = layer(x)
            predictions = x
        
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]


    grads = tape.gradient(class_channel, conv_outputs)


    if grads is None:
        print("Error: Gradients are None. This usually means the model layers are frozen.")
        print("Try running: model.trainable = True before the tape.")
        model.trainable = True
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        img_cv2 = cv2.imread(image_path)
        img_cv2 = cv2.resize(img_cv2, (224, 224))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv2, 0.6, heatmap_color, 0.4, 0)

    filename = "heatmap_" + os.path.basename(image_path)
    output_path = os.path.join("uploads", filename)

    cv2.imwrite(output_path, superimposed_img)

    return filename
