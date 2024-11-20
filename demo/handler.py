import os
import cv2
import sys
import tqdm
import uuid
import shutil
import time
import pickle
import numpy as np
import pandas as pd
import concurrent.futures
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


def write_images_to_db(date, bytes_img, dataset):

	src_img = dataset + '/' + date + '.png'
	img_array = bytes_to_array(bytes_img)
	cv2.imwrite(src_img, img_array)
	return {'status': 'success'}


def write_images(date, img1):

	src_img = './images/' + date + '_img.png'
	cv2.imwrite(src_img, img1)
	return src_img


def get_all_imgs(cust_data):
	imgs = []
	if 'base64_1' in cust_data.keys():
		imgs.append(cust_data['base64_1'])
	if 'base64_2' in cust_data.keys():
		imgs.append(cust_data['base64_2'])
	if 'base64_3' in cust_data.keys():
		imgs.append(cust_data['base64_3'])
	if 'base64_4' in cust_data.keys():
		imgs.append(cust_data['base64_4'])
	if 'base64_5' in cust_data.keys():
		imgs.append(cust_data['base64_5'])
	return imgs


def write_face_features(train_data, model_name='Facenet', enforce_detection=True, detector_backend='retinaface', align=True, normalization='base'):

	t_start = time.time()

	# try:

	representation_filename = str(uuid.uuid1())

	model_names = []
	model_names.append(model_name)

	models = {}
	model = build_model(model_name)
	models[model_name] = model

	custom_model = models[model_name]

	vdb_list = []
	representation_list = []
	for cust_data in train_data['data']:
		vdb_id = {}
		imgs = get_all_imgs(cust_data)
		uuid1 = uuid.uuid1()
		for img in imgs:
			#print(img)
			img = cv2.imread(img)
			#cv2.imwrite("./image.png", img)
			img_representation = represent(img_path = img
								, model_name = model_name, model = custom_model
								, enforce_detection = enforce_detection, detector_backend = detector_backend
								, align = align
								, normalization = normalization
								)
			representation_list.append([uuid1, img_representation])
		vdb_id['cust_id'] = cust_data['customer_id']
		vdb_id['vdb_id'] = uuid1
		vdb_id['vdb_path'] = representation_filename
		vdb_list.append(vdb_id)

	if not os.path.exists('./face_weights/'):
		os.mkdir('./face_weights/')
	repr_file_path = './face_weights/{}.pkl'.format(representation_filename)
	f = open(repr_file_path, "wb")
	pickle.dump(representation_list, f)
	f.close()

	# except:

	#	return {'status': 'failure - please ensure db_path exists or images have a face photo'}

	t_end = time.time()

	return {'status': 'complete', 'time': str(timedelta(seconds=t_end-t_start)), 'vdb_id': vdb_list}


def run_multiprocess(img1, img1_representation, img2_representation, threshold):

	# find distances between embeddings
	distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
										 dst.l2_normalize(img2_representation[1]))
	distance = np.float64(distance)
	if distance <= threshold:
		identified = True
		vdb_id = img2_representation[0]
	else:
		identified = False
		vdb_id = None
	return [img1, 'img2', identified, vdb_id, distance]


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

	img1 = bytes_to_array(bytes_img)

	result = []
	create_image_dir(PATH, LOG)
	# img1 = write_images(date, img_array)

	try:
		custom_model = models[model_name]
		img1_representation = represent(img_path = img1,
										model_name = model_name, model = custom_model,
										enforce_detection = enforce_detection, detector_backend = detector_backend,
										align = align,
										normalization = normalization
										)
		# os.remove(img1)
	except:
		raise ValueError("Something went wrong while reading image")

	face_read_flag = False
	if os.path.exists('./face_weights/'):
		result = []
		for each in os.listdir('./face_weights/'):
			face_repr_file = './face_weights/' + each
			if os.path.exists(face_repr_file):
				f = open(face_repr_file, 'rb')
				representation_list = pickle.load(f)

				# with multiprocessing
				with concurrent.futures.ProcessPoolExecutor() as executor:
					futures = [executor.submit(run_multiprocess, img1, img1_representation, representation_list[i], threshold) for i in range(len(representation_list))]
				counter = 0
				for f in concurrent.futures.as_completed(futures):
					# print(f.result())
					result.append(f.result())
					counter += 1

				face_read_flag = True

	if not face_read_flag:
		return {"verified": False, "comparisons": 0, "time": str(timedelta(seconds=t_end - t_start)), "vdb_id": None, "distance": 1.0}

	res_df = pd.DataFrame(result, columns=['img1', 'img2', 'verified', 'vdb_id', 'score'])
	res_df.drop(['img1', 'img2'], inplace=True, axis=1)
	res_df.to_csv(os.path.join(LOG,date+'_result.csv'), index=False)
	t_end = time.time()
	if True in res_df['verified'].tolist():
		return {"verified": True, "comparisons":res_df.shape[0], "time": str(timedelta(seconds=t_end-t_start)), "vdb_id": min(list(res_df.loc[res_df['verified']==True]['vdb_id'])), "distance": min(list(res_df.loc[res_df['verified']==True]['score']))}
	else:
		return {"verified": False, "comparisons":res_df.shape[0], "time": str(timedelta(seconds=t_end-t_start)), "vdb_id": None, "distance": 1.0}
