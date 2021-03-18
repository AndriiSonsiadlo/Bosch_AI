#Copyright (C) 2020 BardzoFajnyZespol

import json
import shutil
import time
from concurrent.futures.thread import ThreadPoolExecutor
from math import ceil
from threading import Thread
from typing import Dict, Any
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras_preprocessing.image import img_to_array
from kivy.clock import mainthread
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from classes._key_json import *
from classes._learning_config import *
from classes.image import ImageObject
import os
from pathlib import Path
import cv2
import os.path
from classes.model_list import ModelList


class Anomalies(Thread):

	def __init__(self, image_name, autoencoders, autoencoders_info, automode_ref, photo_path):
		Thread.__init__(self)

		self.automode_ref = automode_ref

		self.image_path = ""
		self.photo_path = photo_path
		self.image_name = image_name
		self.temp_image_input = 'input.png'
		self.temp_image_output = 'output.png'
		self.temp_folder = "temp_image_mqtt/"

		self.autoencoders = autoencoders
		self.autoencoders_info = autoencoders_info
		self.thresh: float = 0.0

		self.lstImgObj: dict = {}
		self.imgArrNpByte : dict ={}

		self.status = None

	def run(self):
		try:
			if(len(self.autoencoders_info) != 0):
				exist_image = self.crop_auto()
				if exist_image:
					self.load_images()
					self.find_anomalies()
			else:
				print("Not found correct model")
		except BaseException:
			print("Error")
			pass

	def crop_auto(self):

		def parting(xs, parts):
			part_len = ceil(len(xs) / parts)
			return [xs[part_len * k:part_len * (k + 1)] for k in range(parts)]

		temp_autoencoder_info = []
		for model in self.autoencoders_info:
			temp_autoencoder_info.append(self.autoencoders_info[model])

		shutil.rmtree(self.temp_folder, ignore_errors=True)
		dirname = os.path.dirname(self.temp_folder)
		Path(dirname).mkdir(parents=True, exist_ok=True)

		self.image_name = self.image_name.replace("CheckResult", "")
		img_filename = str(self.image_name) + str('.png')


		if not self.photo_path.endswith("/") or self.photo_path.endswith("\\"):
			self.photo_path = self.photo_path + "\\"
		self.image_path = self.photo_path + img_filename


		# -------------------------------------------------------------------------#
		autoencoders_info = parting(temp_autoencoder_info, parts=6)
		executor = ThreadPoolExecutor(max_workers=6)
		future1 = executor.submit(self.auto_crop_release, autoencoders_info[0], dirname)
		future2 = executor.submit(self.auto_crop_release, autoencoders_info[1], dirname)
		future3 = executor.submit(self.auto_crop_release, autoencoders_info[2], dirname)
		future4 = executor.submit(self.auto_crop_release, autoencoders_info[3], dirname)
		future5 = executor.submit(self.auto_crop_release, autoencoders_info[4], dirname)
		future6 = executor.submit(self.auto_crop_release, autoencoders_info[5], dirname)
		try:
			test = future1.result() + future2.result() + future3.result() + future4.result() + future5.result() + future6.result()
		except BaseException:
			pass

		return True


	def auto_crop_release(self, autoencoders_info, dirname):
		for model in autoencoders_info:
			print(model.name)
			x = model.cropdims['x']
			y = model.cropdims['y']
			w = model.cropdims['w']
			h = model.cropdims['h']

			if not isinstance(x, int):
				x = int(x)
			if not isinstance(y, int):
				y = int(y)
			if not isinstance(w, int):
				w = int(w)
			if not isinstance(h, int):
				h = int(h)


			if self.image_path != '' and w != 0 and h != 0:
				print(self.image_path)
				if self.image_path.endswith(".png") and not self.image_path.startswith("._"):
					print("fileAddress: " + self.image_path)
					image = cv2.imread(self.image_path)
					cropped = image[y:y + h, x:x + w]
					# write the cropped image to disk in PNG format
					cv2.imwrite(f"{dirname}/{model.name}.png", cropped)
					print("Cropped: " + f"{dirname}/{model.name}.png")

		return 1

	def load_images(self):

		fds = os.listdir(self.temp_folder)

		image_paths = []
		# IMAGE_DIMS = self.readImageSize(json_model_path=json_model_path, model_name=model_name)

		for img_file in fds:
			# if (img_file.endswith(f'.png') and not img_file.endswith(self.temp_image_input)
			# 			and not img_file.endswith(self.temp_image_output)):
			for model in self.autoencoders_info:
				if(img_file.startswith(f"{self.autoencoders_info[model].name}.png")):
					pathImg = f'{self.temp_folder}/{img_file}'
					image_paths.append(pathImg)
					break

		for img_path in image_paths:
			image = cv2.imread(img_path, 0)
			image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
			imgArrByte = img_to_array(image)

			# POPRAWIC !!!
			imgObj = ImageObject(path=img_path, byteArray=imgArrByte)
			temp_path = img_path.split('/')
			# print(temp_path)
			file_path = temp_path[2].rsplit('.png', 1)[0]
			# !!!

			imgsArrByte = []
			imgsArrByte.append(imgObj.byteArray)
			self.imgArrNpByte[file_path] = np.array(imgsArrByte) / 255.0
			self.lstImgObj[file_path] = imgObj


	def find_anomalies(self):
		# make predictions on our image data and initialize our list of
		# reconstruction errors
		decoded = {}
		for (autoencoder, imgObj) in zip(sorted(self.autoencoders), sorted(self.lstImgObj.keys())):
			if imgObj.startswith(autoencoder) and imgObj.endswith(autoencoder):
				decoded[autoencoder] = self.autoencoders[autoencoder].predict(self.imgArrNpByte[imgObj])


		# loop over all original images and their corresponding
		# reconstructions
		errors = {}
		for (imgObj, recon) in zip(sorted(self.lstImgObj.keys()), sorted(decoded.keys())):
			# compute the mean squared error between the ground-truth image
			# and the reconstructed image, then add it to our list of errors
			mse = np.mean((self.lstImgObj[imgObj].byteNpArray - decoded[recon]) ** 2)
			errors[imgObj] = mse
			self.lstImgObj[imgObj].mse = mse

		thresh = {}
		for model in self.autoencoders_info:
			thresh[model] = self.autoencoders_info[model].threshold_manual

		idxs_nok = []
		cropdims_nok = {}
		idxs_ok = []
		cropdims_ok = {}

		for (error, model) in zip(errors, self.autoencoders_info):
			print(f"Part: {error}, threshold: {errors[error]}")
			print(f"Model Threshold: {thresh[model]}")
			if (errors[error] > thresh[model]):
				idxs_nok.append(error)
				cropdims_nok[model] = self.autoencoders_info[model].cropdims
			else:
				idxs_ok.append(error)
				cropdims_ok[model] = self.autoencoders_info[model].cropdims

		if (len(idxs_nok) != 0):
			print("Result: NOK")
			self.status = False
			self.automode_ref.show_result(result=False)
		else:
			print("Result: OK")
			self.status = True
			self.automode_ref.show_result(result=True)

		self.automode_ref.show_photo_name(name=self.image_name)
		self.set_photo()


	@mainthread
	def set_photo(self):
		self.automode_ref.show_photo(source=str(self.image_path))


