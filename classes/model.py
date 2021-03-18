#Copyright (C) 2020 BardzoFajnyZespol

from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from math import ceil

import numpy as np
import random
import cv2
from keras.preprocessing.image import img_to_array
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, History, Callback
from classes._key_json import *
from classes._learning_config import *
from classes.convautoencoder import Conv2DAutoencoder
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.optimizers import Adam
import tkinter as tk

# makes image import work on Mac
root = tk.Tk()
root.withdraw()

from tkinter import filedialog
from PIL import Image
from os.path import dirname, abspath
import json
from functions.own_time import *
from functions.add_padding import add_padding

from classes.crop import crop
from functions.get_image_dimensions import get_crop_dims

model_data_dir = dirname(dirname(abspath(__file__))) + "\\model data\\"  # 'Kivy/model data' directory path


class Model:
	def __init__(self, name: str = "unnamed", author: str = "Unknown", comment: str = ""):
		self.name = name
		# self.created = datetime.now()
		self.created = "date error"
		self.author = author
		self.comment = comment
		self.selected = False
		self.ok_photos = ['none selected']
		self.nok_photos = ['none selected']
		self.data_path = 'model data/'
		self.epochs = 0
		self.batch_size = 0
		self.learning_time = 0.0
		self.threshold = 0.0
		self.threshold_manual = self.threshold
		self.cropdims = {"x": 0, "y": 0, "w": 0, "h": 0}
		self.automode_selected = False

	def get_time_created(self):  # return date and time
		return self.created

	# return str(self.created.strftime("%d.%m.%Y, %X"))

	def edit(self, new_name: str):
		self.name = new_name

	# what should be editable? what about creation date?
	def crop(self, screen, photo_paths):

		def parting(xs, parts):
			part_len = ceil(len(xs) / parts)
			return [xs[part_len * k:part_len * (k + 1)] for k in range(parts)]

		cropped = 0
		x, y, w, h = 0, 0, 0, 0
		crop_dir = os.path.join(os.path.dirname(photo_paths[0]), 'cropped')
		if os.path.exists(crop_dir):
			for f in os.listdir(crop_dir):
				file_path = os.path.abspath(os.path.join(crop_dir, f))
				os.remove(file_path)

		if screen.ids.crop_checkbox.active:
			if screen.ids.crop_x_input.text == '' and screen.ids.crop_y_input.text == '' and screen.ids.crop_w_input.text == '' and screen.ids.crop_h_input.text == '':
				x, y, w, h = get_crop_dims(photo_paths[0])
			else:
				x = int(screen.ids.crop_x_input.text)
				y = int(screen.ids.crop_y_input.text)
				w = int(screen.ids.crop_w_input.text)
				h = int(screen.ids.crop_h_input.text)

			parts_image_path = parting(photo_paths, parts=8)
			executor = ThreadPoolExecutor(max_workers=8)
			future1 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[0], cropped)
			future2 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[1], cropped)
			future3 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[2], cropped)
			future4 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[3], cropped)
			future5 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[4], cropped)
			future6 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[5], cropped)
			future7 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[6], cropped)
			future8 = executor.submit(self.crop_release, x, y, w, h, screen, parts_image_path[7], cropped)
			try:
				test = future1.result() + future2.result() + future3.result() + future4.result() + future5.result() + future6.result() + future7.result() + future8.result()
			except BaseException:
				pass

			cropped = 1
		return cropped

	def crop_release(self, x, y, w, h, screen, photo_paths, cropped):
		for img in photo_paths:
			if img.endswith('.png') and not img.startswith("._") and cropped != 1:  # bulletproofing against thumbnails
				print(img)
				# x, y, w, h = get_crop_dims(img)
				screen.ids.crop_x_input.text = str(x)
				screen.ids.crop_y_input.text = str(y)
				screen.ids.crop_w_input.text = str(w)
				screen.ids.crop_h_input.text = str(h)
				crop(x, y, w, h, img)
		return 1

	def get_photos_from_drop(self, screen, ok=True):
		if ok:
			dir_path = str(screen.ids.import_ok_btn.path)
		else:
			dir_path = str(screen.ids.import_nok_btn.path)
		if os.path.isdir(dir_path):
			files_names = os.listdir(dir_path)
			files_absolute = []
			for f in files_names:
				f = os.path.abspath(os.path.join(dir_path, f))
				files_absolute.append(f)
			photos = files_absolute
			if len(photos) > 0:
				if ok:
					self.ok_photos.clear()
				else:
					self.nok_photos.clear()
				if 0 not in self.cropdims.values():  # crop if there is input in x,y,w,h boxes
					crop(self.cropdims["x"], self.cropdims["y"], self.cropdims["w"], self.cropdims["h"],
					     dir_path)
				cropped = 0
				if screen.ids.crop_checkbox.active:
					cropped = self.crop(screen, photos)
				if cropped:
					dir_path = os.path.join(dir_path, 'cropped')
					photo_names = os.listdir(dir_path)
					photo_paths = []
					for f in photo_names:
						photo_paths.append(os.path.abspath(os.path.join(dir_path, f)))
				if len(photos) == 1:
					print("One element in directory. Not loaded.")
					return -1
				for img in os.listdir(dir_path):
					img = os.path.abspath(os.path.join(dir_path, img))
					if img.endswith(('.png')) and not img.startswith("._"):  # bulletproofing against thumbnails
						if ok:
							self.ok_photos.append(img)
							num_ok_loaded = len(self.ok_photos)
							other_loaded = screen.new_model.nok_photos
							screen.ids.num_ok_files.text = str(num_ok_loaded) + " loaded"
							screen.ids.num_ok_files.opacity = 1
							screen.ids.ok_dir_icon.opacity = 0
						else:
							self.nok_photos.append(img)
							num_nok_loaded = len(self.nok_photos)
							other_loaded = screen.new_model.ok_photos
							screen.ids.num_nok_files.text = str(num_nok_loaded) + " loaded"
							screen.ids.num_nok_files.opacity = 1
							screen.ids.nok_dir_icon.opacity = 0
						if len(other_loaded) > 1:
							screen.enable_learning_btn()

	def get_photos(self, screen, ok=True):  # True - ok dataset, False - nok dataset

		root = tk.Tk()
		root.withdraw()
		photo_paths = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])
		num_loaded = 0
		if photo_paths:
			if ok:
				self.ok_photos.clear()
			else:
				self.nok_photos.clear()
			dir_path = os.path.dirname(photo_paths[0])

			if 0 not in self.cropdims.values():  # crop if there is input in x,y,w,h boxes
				crop(self.cropdims["x"], self.cropdims["y"], self.cropdims["w"], self.cropdims["h"],
				     dir_path)  # crop by box input
			cropped = 0

			if screen.ids.crop_checkbox.active:
				cropped = self.crop(screen, photo_paths)

			if cropped:
				dir_path = os.path.join(dir_path, 'cropped')
				photo_names = os.listdir(dir_path)
				photo_paths = []
				for f in photo_names:
					photo_paths.append(os.path.abspath(os.path.join(dir_path, f)))

			if len(photo_paths) == 1:
				print("One element in directory. Not loaded.")
				return -1
			elif len(photo_paths) > 1:
				screen.ids.cropped_area.source = photo_paths[0]
			for img in photo_paths:
				if img.endswith(('.png')) and not img.startswith("._"):  # bulletproofing against thumbnails
					if ok:
						self.ok_photos.append(img)
						num_loaded = len(self.ok_photos)
					else:
						self.nok_photos.append(img)
						num_loaded = len(self.nok_photos)
		return num_loaded

	def save_ok_sample(self, img_path):
		img = Image.open(img_path)
		img = img.resize((100, 100))
		file_path = self.data_path + '/ok_sample.jpg'
		img.save(file_path)

	def load_images(self):

		# initialize the data and labels
		imgsArrByte = []
		imgsLabel = []
		# loop over the input OK and NOK images
		for imgPath in self.ok_photos:
			# load the image, pre-process it, and store it in the data list
			image = cv2.imread(imgPath, 0)
			# print("image padding: " + str(imgPath))
			# image = add_padding(image, IMAGE_DIMS[0], IMAGE_DIMS[1]) #resizes to img_dims maintaining aspect ratio
			image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
			image = img_to_array(image)
			imgsArrByte.append(image)

			# update the labels list # 1 = OK, Valid
			label = 1
			imgsLabel.append(label)

		for imgPath in self.nok_photos:
			# load the image, pre-process it, and store it in the data list
			image = cv2.imread(imgPath, 0)
			image = add_padding(image, IMAGE_DIMS[0], IMAGE_DIMS[1])  # resizes to img_dims maintaining aspect ratio
			# image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
			image = img_to_array(image)
			imgsArrByte.append(image)

			# update the labels list # 0 = NOK, Anomaly
			label = 0
			imgsLabel.append(label)

		# scale the raw pixel intensities to the range [0, 1]
		imgsArrByte = np.array(imgsArrByte, dtype="float") / 255.0
		imgsLabel = np.array(imgsLabel)

		print("[INFO] data matrix: {:.2f}MB".format(imgsArrByte.nbytes / (1024 * 1000.0)))

		return (imgsArrByte, imgsLabel)

	def create_unsupervised_dataset(self, data, labels, validLabel: int = 1,
	                                anomalyLabel: int = 0,
	                                seed=42, contamValid: float = 1.0):

		# grab all indexes of the supplied class label that are *truly*
		# that particular label, then grab the indexes of the image
		# labels that will serve as our "anomalies"
		validIdxs = np.where(labels == validLabel)[0]
		anomalyIdxs = np.where(labels == anomalyLabel)[0]

		# randomly shuffle both sets of indexes
		random.shuffle(validIdxs)
		random.shuffle(anomalyIdxs)

		# compute the total number of anomaly data points to select
		i = int(len(validIdxs) * contamAnomaly)
		anomalyIdxs = anomalyIdxs[:i]

		i = int(len(validIdxs) * contamValid)
		validIdxs = validIdxs[:i]

		# use NumPy array indexing to extract both the valid images and
		# "anomlay" images
		validImages = data[validIdxs]
		anomalyImages = data[anomalyIdxs]

		# stack the valid images and anomaly images together to form a
		# single data matrix and then shuffle the rows
		images = np.vstack([validImages, anomalyImages])
		np.random.seed(seed)
		np.random.shuffle(images)

		# return the set of images
		return images

	def on_train_batch_begin(self, batch, logs=None):
		print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

	def begin_learning(self, epochs: int = 2000, batch_size: int = 48,
	                   contamValid: float = 1., early_stopping=True):

		import time
		modelPath = self.data_path + '/model_file.model'
		plotPath = self.data_path + '/plot.png'
		jsonPath = self.data_path + '/model_data.json'

		# initialize the number of epochs to train for, initial learning rate,
		# and batch size

		self.epochs = epochs
		self.batch_size = batch_size

		# grab the image paths and randomly shuffle them
		print("[INFO] loading images...")
		(imgsArrByte, imgsLabel) = self.load_images()

		# build our unsupervised dataset of images with a small amount of
		# contamination (i.e., anomalies) added into it
		print("[INFO] creating unsupervised dataset...")
		images = self.create_unsupervised_dataset(imgsArrByte, imgsLabel, validLabel=1, anomalyLabel=0,
		                                          contamValid=contamValid)

		# construct the training and testing split
		(trainX, testX) = train_test_split(images, test_size=test_size, random_state=42)

		learningTimeStart = time.time()

		# construct our convolutional autoencoder
		print("[INFO] building autoencoder...")
		(encoder, decoder, autoencoder) = Conv2DAutoencoder.build(IMAGE_DIMS[0], IMAGE_DIMS[1],
		                                                          IMAGE_DIMS[2])  # (28, 28, 1)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / self.epochs)
		autoencoder.compile(loss="mse", optimizer=opt)

		# train the convolutional autoencoder
		H = History()
		if early_stopping == True:
			callbacks = [H, EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)]
		else:
			callbacks = None
		H = autoencoder.fit(
			trainX, trainX,
			validation_data=(testX, testX),
			epochs=self.epochs,
			batch_size=self.batch_size,
			callbacks=callbacks,
		)

		learningTimeStop = time.time() - learningTimeStart
		self.learning_time = learningTimeStop
		epochs = len(H.history['loss'])
		self.epochs = epochs
		# construct a plot that plots and saves the training history
		N = np.arange(0, epochs)
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(N, H.history["loss"], label="train_loss")
		plt.plot(N, H.history["val_loss"], label="val_loss")
		plt.title("Training Loss")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss")
		plt.legend(loc="lower left")
		plt.savefig(plotPath)

		# serialize the autoencoder model to disk
		print("[INFO] saving autoencoder...")
		autoencoder.save(modelPath, save_format="h5")
		decoded = autoencoder.predict(images)

		errors = []
		# loop over all original images and their corresponding
		# reconstructions
		for (image, recon) in zip(images, decoded):
			# compute the mean squared error between the ground-truth image
			# and the reconstructed image, then add it to our list of errors
			mse = np.mean((image - recon) ** 2)
			errors.append(mse)

		# compute the q-th quantile of the errors which serves as our
		# threshold to identify anomalies -- any data point that our model
		# reconstructed with > threshold error will be marked as an outlier
		thresh = np.quantile(errors, quantile)
		self.threshold = thresh
		self.threshold_manual = self.threshold
		self.created = getTime("day") + '.' + getTime("month") + '.' + getTime("year") + ' ' + getTime(
			"hour") + ':' + getTime("minute") + ':' + getTime("second")
		print("[INFO] mse threshold: {}".format(thresh))
		print(f"[INFO] learning time: {learningTimeStop}")

		dataJSON = {
			information_k: {
				model_name_k: self.name,
				author_k: self.author,
				comment_k: self.comment,
				d_m_y_k: getTime("day") + '.' + getTime("month") + '.' + getTime("year"),
				h_m_s_k: getTime("hour") + ':' + getTime("minute") + ':' + getTime("second"),
				learning_time_k: round(learningTimeStop, 2),
				ok_photos_k: self.ok_photos,
				nok_photos_k: self.nok_photos,
				automode_k: self.automode_selected
			},

			data_k: {
				thresh_auto_k: thresh,
				thresh_manual_k: thresh,
				epochs_k: epochs,
				test_size_k: test_size,
				train_size_k: (1.0 - test_size),
				contam_anomaly_k: contamAnomaly,
				batch_size_k: batch_size,
				resized_size_image_h_k: IMAGE_DIMS[1],
				resized_size_image_w_k: IMAGE_DIMS[0],
				train_dataset_k: len(trainX),
				test_dataset_k: len(testX),
				crop_dimenstion_k: {
					x_k: self.cropdims["x"],
					y_k: self.cropdims["y"],
					w_k: self.cropdims["w"],
					h_k: self.cropdims["h"]
				}
			}
		}

		# writing to .json file
		print("[INFO] saving data of model to .json...")
		with open(jsonPath, "w") as write_file:
			json.dump(dataJSON, write_file)
			json.dumps(dataJSON, indent=4)
