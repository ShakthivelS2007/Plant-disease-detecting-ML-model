import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os


DATASET_PATH = r"C:\Users\SAKTHIVEL.S\plant-hack\dataset\train" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,         
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3], 
    fill_mode='reflect',
    validation_split=0.2    
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True            
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(3, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print("Training started... ")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)


model.save('tomato_model_v3_field_ready.h5')
print("\nSuccess! Model saved as 'tomato_model_v3_field_ready.h5'")
