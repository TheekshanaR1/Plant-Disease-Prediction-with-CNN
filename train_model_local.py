# -*- coding: utf-8 -*-
"""
Plant Disease Prediction with CNN - Local Training Script
Modified for local execution without Kaggle/Colab dependencies
"""

# Set seeds for reproducibility
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

# Importing the dependencies
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset Path - You need to download and extract the PlantVillage dataset
# Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Or use the dataset that's already on your system

# For this script, we'll assume you have the dataset in the 'plantvillage_dataset' folder
# Please update this path if your dataset is located elsewhere
base_dir = 'plantvillage_dataset/color'

# Check if dataset exists
if not os.path.exists(base_dir):
    print(f"\n⚠️ ERROR: Dataset not found at '{base_dir}'")
    print("\nPlease download the PlantVillage dataset:")
    print("1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("2. Download and extract it to this directory")
    print("3. Make sure the folder structure is: plantvillage_dataset/color/")
    print("\nAlternatively, you can update the 'base_dir' variable in this script.")
    exit(1)

print(f"\n✓ Dataset found at: {base_dir}")
print(f"Number of plant disease classes: {len(os.listdir(base_dir))}")

# Image Parameters
img_size = 224
batch_size = 32

print(f"\nImage size: {img_size}x{img_size}")
print(f"Batch size: {batch_size}")

# Image Data Generators
print("\n📊 Creating data generators...")
data_gen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values from [0, 255] to [0, 1]
    validation_split=0.2          # Reserve 20% of the data for validation
)

# Train Generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical',
    shuffle=True
)

# Validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

print(f"\n✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"✓ Number of classes: {train_generator.num_classes}")

# Model Definition
print("\n🏗️ Building CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Model summary
model.summary()

# Compile the Model
print("\n⚙️ Compiling model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the Model
print("\n🚀 Starting model training...")
print("This may take a while depending on your hardware...")

epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=1
)

# Model Evaluation
print("\n📈 Evaluating model...")
val_loss, val_accuracy = model.evaluate(
    validation_generator, 
    steps=validation_generator.samples // batch_size
)
print(f"\n✓ Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"✓ Validation Loss: {val_loss:.4f}")

# Plot training & validation accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("\n✓ Training history saved as 'training_history.png'")
plt.show()

# Create a mapping from class indices to class names
class_indices = {v: k for k, v in train_generator.class_indices.items()}

# Save the class names as json file
print("\n💾 Saving class indices...")
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("✓ Class indices saved as 'class_indices.json'")

# Save the model
model_save_path = 'app/trained_model/plant_disease_prediction_model.h5'
os.makedirs('app/trained_model', exist_ok=True)

print(f"\n💾 Saving model to '{model_save_path}'...")
model.save(model_save_path)
print(f"✓ Model saved successfully!")

# Also save in the root directory for backup
backup_path = 'plant_disease_prediction_model.h5'
model.save(backup_path)
print(f"✓ Backup model saved as '{backup_path}'")

# Copy class_indices.json to app folder
import shutil
shutil.copy('class_indices.json', 'app/class_indices.json')
print("✓ Class indices copied to app folder")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Final Results:")
print(f"   - Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"   - Total Classes: {train_generator.num_classes}")
print(f"   - Epochs Trained: {epochs}")
print(f"\n📁 Generated Files:")
print(f"   - Model: {model_save_path}")
print(f"   - Class indices: app/class_indices.json")
print(f"   - Training history: training_history.png")
print(f"\n✨ Next Steps:")
print(f"   Run the web app with: streamlit run app/main.py")
print("="*60)

# Test prediction function
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Test with sample images if they exist
print("\n🧪 Testing predictions on sample images...")
test_images_dir = 'test_images'
if os.path.exists(test_images_dir):
    test_images = os.listdir(test_images_dir)
    for img_name in test_images[:3]:  # Test first 3 images
        img_path = os.path.join(test_images_dir, img_name)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                predicted_class = predict_image_class(model, img_path, class_indices)
                print(f"   {img_name}: {predicted_class}")
            except Exception as e:
                print(f"   Error processing {img_name}: {e}")
else:
    print("   No test images found in 'test_images' folder")

print("\n✅ All done!")
