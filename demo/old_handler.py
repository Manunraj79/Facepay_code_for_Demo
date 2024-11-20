import os
import cv2
import shutil
import time
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from io import BytesIO
from demo import distance as dst
from demo.verify import initialize_input, build_model, represent


def create_image_dir(PATH, LOG):

	if not os.path.isdir(PATH):
		os.mkdir(PATH)
	if not os.path.isdir(LOG):
		os.mkdir(LOG)


def bytes_to_array(b: bytes) -> np.ndarray:

	np_bytes = BytesIO(b)
	return np.load(np_bytes, allow_pickle=True)


def write_images(date, img1):

	src_img = './images/' + date + '_img.png'
	cv2.imwrite(src_img, img1)
	return src_img


def write_face_features(db_path, model_name='Facenet', enforce_detection=True, detector_backend='retinaface', align=True, normalization='base'):

	t_start = time.time()

	try:

		model_names = []
		model_names.append(model_name)

		models = {}
		model = build_model(model_name)
		models[model_name] = model

		custom_model = models[model_name]

		representation_list = []
		for img in os.listdir(db_path):
			img_path = db_path + '/' + img
			img_representation = represent(img_path = img_path
								, model_name = model_name, model = custom_model
								, enforce_detection = enforce_detection, detector_backend = detector_backend
								, align = align
								, normalization = normalization
								)
			representation_list.append(img_representation)

		f = open('./model_weights/representation.pkl', "wb")
		pickle.dump(representation_list, f)
		f.close()

	except:

		return {'status': 'failure - please ensure db_path exists or images have a face photo'}

	t_end = time.time()

	return {'status': 'complete', 'time': str(timedelta(seconds=t_end-t_start))}


def validate_user(date, bytes_img, dataset, threshold=0.85, PATH='./images', LOG='./log_history'):

	t_start = time.time()
	model_name = 'Facenet'
	distance_metric = 'euclidean_l2'
	detector_backend = 'retinaface'
	normalization = 'base'
	align = True
	enforce_detection = True

	model_names, metrics = [], []
	model_names.append(model_name)
	metrics.append(distance_metric)

	models = {}
	model = build_model(model_name)
	models[model_name] = model

	img_array = bytes_to_array(bytes_img)

	result = []
	create_image_dir(PATH, LOG)
	img1 = write_images(date, img_array)
	print(img1)
	# try:
	custom_model = models[model_name]
	img1_representation = represent(img_path = img1,
									model_name = model_name, model = custom_model,
									enforce_detection = enforce_detection, detector_backend = detector_backend,
									align = align,
									normalization = normalization
									)
	#os.remove(img1)
	# except:
	# 	raise ValueError("img_path does not exist or model not defined")

	if os.path.exists("./model_weights/representation.pkl"):

		result = []
		f = open('./model_weights/representation.pkl', 'rb')
		representation_list = pickle.load(f)

		for img2_representation in representation_list:
			#find distances between embeddings
			distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
			distance = np.float64(distance)
			if distance <= threshold:
				identified = True
			else:
				identified = False
			result.append([img1, 'img2', identified, distance])

	else:
		for each in os.listdir(dataset):
			img2 = dataset + '/' + each

			if img1 != img2:

				img1_path, img2_path = img1, img2

				img_list = initialize_input(img1_path, img2_path)
				instance = img_list[0]

				custom_model = models[model_name]

				if type(instance) == list and len(instance) >= 2:
					img2_path = instance[1]

					img2_representation = represent(img_path = img2_path
							, model_name = model_name, model = custom_model
							, enforce_detection = enforce_detection, detector_backend = detector_backend
							, align = align
							, normalization = normalization
							)

					#find distances between embeddings
					distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
					distance = np.float64(distance)

					if distance <= threshold:
						identified = True
					else:
						identified = False

					result.append([img1, img2, identified, distance])

	res_df = pd.DataFrame(result, columns=['img1', 'img2', 'verified', 'score'])
	res_df.to_csv(os.path.join(LOG,date+'_result.csv'), index=False)
	t_end = time.time()
	if True in res_df['verified'].tolist():
		return {"verified": True, "comparisons":res_df.shape[0], "time": str(timedelta(seconds=t_end-t_start)), "distance": min(list(res_df.loc[res_df['verified']==True]['score']))}
	else:
		return {"verified": False, "comparisons":res_df.shape[0], "time": str(timedelta(seconds=t_end-t_start)), "distance": 1.0}
