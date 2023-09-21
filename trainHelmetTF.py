import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os

# Set the data directory containing your training images
data_dir = 'data/train'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")
print(f"Directory '{data_dir}' does exist!")

# Define data generator with transformations
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a custom dataset generator
batch_size = 64
class_mode = 'categorical'  # For classification
num_classes = len(os.listdir(data_dir))
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=True
)

# Define and compile the model
base_model = keras.applications.ResNet50(
    include_top=False,  # Exclude the fully connected layer
    weights='imagenet',  # Use pre-trained weights
    input_shape=(224, 224, 3)
)

# Create a new output layer with the correct number of units
output_units = num_classes
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(output_units, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
num_epochs = 10  # Adjust the number of training epochs as needed
model.fit(train_generator, epochs=num_epochs)

# Save the trained model
model.save('helmet_classifier_tf.h5')
