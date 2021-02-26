# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
from skimage import io
import PIL
import os
import cv2
from google.cloud import storage

# Flask App Engine 
# Define a Flask app
app = Flask(__name__)

CLOUD_STORAGE_BUCKET = 'oct-classification-290613.appspot.com'

# Prepare Keras Model
# Model files
MODEL_ARCHITECTURE = './model/custom_vgg.json'
MODEL_WEIGHTS = './model/custom_vgg_weights.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''
	img = io.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	IMG_ = cv2.resize(img,(224,224),3)

	IMG_ = np.asarray(IMG_)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = IMG_.reshape(1, 224, 224, 3)


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	prediction = model.predict_classes(IMG_)

	return prediction


# FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'TRAIN': ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Normal'],
	           'VALIDATION': ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Normal'],
	           'TEST': ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Normal']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files.get('file')

		if not f:
			return 'No file uploaded.'
		
		# gcs = storage.Client.from_service_account_json('oct-classification-39e9bacfee27.json') for testing in local environment

		# Initialise Cloud storage
		gcs = storage.Client()
		
		# Get Bucket for uploading images
		bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
		
		# Saving Image to bucket
		blob = bucket.blob(f.filename)
		blob.upload_from_string(
			f.read(),
			content_type=f.content_type
		)
		
		# Get Image URL for processing
		file_path = blob.public_url

		# Make a prediction
		prediction = model_predict(file_path, model)
		predicted_class = classes['TRAIN'][prediction[0]]
		print('We think that is {}.'.format(predicted_class.lower()))

		return str(predicted_class).lower()

if __name__ == '__main__':
	app.run(debug = True)