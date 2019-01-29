# all data science libs
from functions import *

from sklearn.datasets import load_files       

from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
    
import tensorflow as tf

from tqdm import tqdm
import numpy as np
from glob import glob
import time
import cv2                
import matplotlib.pyplot as plt

# all web libs
import json
import plotly
import pandas as pd

import os

from flask import Flask
from flask import render_template, request, jsonify, flash, redirect, url_for
from plotly.graph_objs import Bar

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# init Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# build model in selected architecture
TL_model = Sequential()
TL_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
TL_model.add(Dense(133, activation='softmax'))

# load best saved model
TL_model.load_weights('weights.best.InceptionV3.hdf5')

# to recognize humans
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# to recognize dogs
# why to use global here? https://github.com/jrosebr1/simple-keras-rest-api/issues/1
global ResNet50_model
ResNet50_model = ResNet50(weights='imagenet')

# to predict breed
global InceptionV3_model
InceptionV3_model = InceptionV3(weights='imagenet', include_top=False)

global graph
graph = tf.get_default_graph()

# all dog names as array
dog_names = get_dog_names()

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    with graph.as_default():
        pred = ResNet50_model.predict(img)
        prediction = np.argmax(pred)
        return ((prediction <= 268) & (prediction >= 151)) 
    
    return False
    

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def predict_breed(img_path):
    with graph.as_default():
        # Extract the bottleneck features for Resnet CNN model
        InceptionV3_tensor = get_InceptionV3_tensor(path_to_tensor(img_path))
        bottleneck_feature = InceptionV3_model.predict(InceptionV3_tensor)
    
        # obtain predicted vector
        predicted_vector = TL_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]
    return "no name found"

def detect_breed(image_path):
    true_detection = True
    is_human = face_detector(image_path)
    is_dog = dog_detector(image_path)
    
    predicted_class = ""
    predicted_breed = ""
    
    if is_human and is_dog:
        predicted_class = "Error: There was a dog and a human face detected on the image."
        true_detection = False
    elif is_human:
        predicted_class = "A human face was detected on the image!"
    elif is_dog:
        predicted_class = "A dog was detected on the image!"
    else:
        predicted_class = "Error: There was neither a human face or a dog detected on the image."
        true_detection = False
     
    if true_detection:
        breed = predict_breed(image_path)
        if is_human:
            predicted_breed = "You look like a " + breed
        else:
            predicted_breed = "The dog's breed is " + breed
                
    return predicted_class, predicted_breed



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go',  methods=['GET', 'POST'])
def go():
    # save user input in query
    query = request.args.get('input-image', '') 

    if request.method == 'POST':
        # check if the post request has the file part
        if 'input-image' not in request.files:
            flash('No file part')
            return render_template('master.html')
        
        file = request.files['input-image']

        if file.filename == '':
            flash('No selected file')
            return render_template('master.html')
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = app.config['UPLOAD_FOLDER'] + '/' + filename
            
            # predict
            predicted_class, predicted_breed = detect_breed(image_path)
            return render_template(
                'go.html',
                query=query,
                img_path=image_path,
                predicted_class=predicted_class,
                predicted_breed=predicted_breed
            )

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()