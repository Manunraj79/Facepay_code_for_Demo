import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
import numpy as np
from demo import Facenet, functions
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def initialize_input(img1_path, img2_path = None):

	if ((type(img2_path) == str and img2_path != None) or (isinstance(img2_path, np.ndarray) and img2_path.any())):
		img_list = [[img1_path, img2_path]]
	else:
		img_list = [img1_path]

	return img_list


def build_model(model_name):

	model_obj = {}
	model = Facenet.loadModel

	if model:
		model = model()
		model_obj[model_name] = model

	return model_obj[model_name]


def represent(img_path, model_name = 'Facenet', model = None, enforce_detection = True, detector_backend = 'retinaface', align = True, normalization = 'base'):

	if model is None:
		model = build_model(model_name)

	#decide input shape
	input_shape_x, input_shape_y = functions.find_input_shape(model)

	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, align = align)

	#custom normalization
	img = functions.normalize_input(img = img, normalization = normalization)

	embedding = model.predict(img)[0].tolist()

	return embedding
