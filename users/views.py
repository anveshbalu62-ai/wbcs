from django.conf import settings
from django.shortcuts import render, redirect
import pandas as pd
from .models import UserRegistrationModel
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model # type: ignore

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 

 
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import os
import itertools
import warnings
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)

import threading

MODEL_PATH = os.path.join(settings.BASE_DIR, 'multiclass.h5')
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Global training status
# Global training status
_TRAINING_STATUS = {
    'is_running': False,
    'last_history': {
        'accuracy': [],
        'loss': [],
        'val_loss': []
    }
}

def training_thread():
    global _TRAINING_STATUS
    _TRAINING_STATUS['is_running'] = True

    try:
        img_size = (224, 224)
        train_data_dir = os.path.join('media', 'main_dataset', 'train')
        val_data_dir = os.path.join('media', 'main_dataset', 'validation')

        batch_size = 32
        epochs = 5

        # Generators
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_batches = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_batches = val_datagen.flow_from_directory(
            val_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(4, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]

        # 🔥 INITIAL TRAINING
        history1 = model.fit(
            train_batches,
            epochs=epochs,
            validation_data=val_batches,
            callbacks=callbacks
        )

        # 🔥 FINE TUNING
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_batches,
            epochs=5,
            validation_data=val_batches,
            callbacks=callbacks
        )

        # 🔥 COMBINE HISTORIES
        full_acc = history1.history['accuracy'] + history2.history['accuracy']
        full_loss = history1.history['loss'] + history2.history['loss']
        full_val_loss = history1.history['val_loss'] + history2.history['val_loss']

        _TRAINING_STATUS['last_history'] = {
            'accuracy': full_acc,
            'loss': full_loss,
            'val_loss': full_val_loss
        }

        # 🔥 SAVE MODEL
        model.save(MODEL_PATH)

        # 🔥 PLOTS
        acc_path = os.path.join(settings.MEDIA_ROOT, 'accuracy_plot.png')
        loss_path = os.path.join(settings.MEDIA_ROOT, 'loss_plot.png')

        plt.figure()
        plt.plot(full_acc)
        plt.title('Accuracy')
        plt.savefig(acc_path)
        plt.close()

        plt.figure()
        plt.plot(full_loss)
        plt.title('Loss')
        plt.savefig(loss_path)
        plt.close()

    except Exception as e:
        print("Training error:", e)

    finally:
        _TRAINING_STATUS['is_running'] = False


def training(request):
    global _TRAINING_STATUS

    if not _TRAINING_STATUS['is_running']:
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        messages.success(request, "Training started!")

    history = _TRAINING_STATUS.get('last_history', {})

    accs = history.get('accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    return render(request, 'users/training.html', {
        'accuracy': accs if accs else [0],
        'loss': loss if loss else [0],
        'val_loss': val_loss if val_loss else [0],
        'is_training': _TRAINING_STATUS['is_running']
    })
 
# Global model variable for caching
_MODEL = None
_MODEL_MTIME = None

def get_model():
    global _MODEL, _MODEL_MTIME
    if os.path.exists(MODEL_PATH):
        current_mtime = os.path.getmtime(MODEL_PATH)
        if _MODEL is None or _MODEL_MTIME != current_mtime:
            print(f"Loading/Reloading model from {MODEL_PATH}...")
            _MODEL = load_model(MODEL_PATH)
            _MODEL_MTIME = current_mtime
    else:
        print(f"Model file not found at {MODEL_PATH}")
        _MODEL = None
        _MODEL_MTIME = None
    return _MODEL

def predictions(request):
    predicted_class = None
    file_url = None
    confidence_scores = None
    
    model = get_model()
    if model is None:
        messages.error(request, "Model file not found. Please go to the Training page and click 'Train Model' first.")
        return redirect('UserHome')

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)

        # Preprocess image using same pipeline as training
        img_path = os.path.join(fs.location, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        try:
            prediction = model.predict(img_array)
        except ValueError as e:
            if "incompatible with the layer" in str(e):
                messages.error(request, "The existing model is outdated (expected smaller images). Please go to the Training page and click 'Train Model' to upgrade.")
                return redirect('UserHome')
            raise e
        predicted_class = class_names[np.argmax(prediction)]

        # Build confidence scores for all classes
        confidence_scores = [
            {'name': class_names[i], 'score': round(float(prediction[0][i]) * 100, 2)}
            for i in range(len(class_names))
        ]

    return render(request, 'users/detection.html', {
        'predicted_class': predicted_class,
        'image_url': file_url,
        'confidence_scores': confidence_scores,
    })

