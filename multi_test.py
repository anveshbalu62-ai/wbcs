import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def get_accuracy(data_dir, model):
    results = {c: {'correct': 0, 'total': 0, 'predictions': {}} for c in class_names}
    
    for cls in class_names:
        dir_path = os.path.join(data_dir, cls)
        if not os.path.exists(dir_path):
            continue
            
        imgs = os.listdir(dir_path)
        # Limit to 100 images per class for speed
        imgs = imgs[:100]
        
        for img_name in imgs:
            img_path = os.path.join(dir_path, img_name)
            
            # Use the EXACT preprocessing from the new training script
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            pred = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(pred)
            predicted_class = class_names[predicted_idx]
            
            results[cls]['total'] += 1
            if predicted_class == cls:
                results[cls]['correct'] += 1
            
            results[cls]['predictions'][predicted_class] = results[cls]['predictions'].get(predicted_class, 0) + 1
            
    return results

if __name__ == "__main__":
    print("Loading model...")
    model = load_model('multiclass.h5')
    
    print("Starting diagnostics...")
    test_results = get_accuracy(r'media\main_dataset\test', model)
    train_results = get_accuracy(r'media\main_dataset\train', model)

    final_report = {
        'test': test_results,
        'train': train_results
    }

    with open('diagnostics.json', 'w') as f:
        json.dump(final_report, f, indent=4)

    print("Diagnostics complete. Results saved to diagnostics.json")
