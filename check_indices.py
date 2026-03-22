import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, 'media', 'main_dataset', 'train')

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Class Indices: {train_generator.class_indices}")
