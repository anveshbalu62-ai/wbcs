import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_prediction(img_path, expected_class):
    model = load_model('multiclass.h5')
    class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
        
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(f"[{expected_class}] prediction: {predicted_class} ({confidence:.2f}%)")
    return predicted_class == expected_class

if __name__ == "__main__":
    base_dir = r"media\main_dataset\train"
    for cls in ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']:
        dir_path = os.path.join(base_dir, cls)
        try:
            sample_img = os.listdir(dir_path)[0]
            img_path = os.path.join(dir_path, sample_img)
            get_prediction(img_path, cls)
        except Exception as e:
            print(f"Error checking {cls}: {e}")
