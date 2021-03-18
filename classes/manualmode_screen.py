#Copyright (C) 2020 BardzoFajnyZespol

import shutil
import threading
import time
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from kivy.uix.screenmanager import Screen
import matplotlib.pyplot as plt

from classes._key_json import *
from classes.image import ImageObject
from classes.model_list import ModelList
import tkinter as tk
from tkinter import filedialog
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
import json
import os

import csv
from datetime import datetime

class ManualMode(Screen):
	model_name = "N/A"
	created = "N/A"
	author = "N/A"
	comment = "N/A"
	loaded_photos = []
	preview_index = 0
	autoencoderName = "N/A"
	autoencoder = None
	temp_folder = "temp_manualmode_crop/"

	def __init__(self, **kw):
		super().__init__(**kw)
		self.model_list = ModelList()

	def load_list(self):
		self.model_list = ModelList()

	# display model info on screen
	def set_model_data(self, name):
		model = self.model_list.find_first(name)
		self.ids.model_name.text = model.name
		self.ids.created_date.text = model.get_time_created()
		self.ids.author.text = model.author
		self.ids.comment.text = model.comment
		self.ids.num_iterations.text = str(model.epochs)
		self.ids.batch_size.text = str(model.batch_size)
		self.ids.num_ok.text = str(len(model.ok_photos))
		self.ids.num_nok.text = str(len(model.nok_photos))

		if model.threshold_manual == model.threshold:
			self.ids.threshold.text = str("{:.5f}".format(model.threshold)) + '(auto)'
		else:
			self.ids.threshold.text = str("{:.5f}".format(model.threshold_manual)) + '(manual)'

		self.ids.crop_xywh.text = 'x: ' + str(model.cropdims["x"]) + ', y: ' + str(model.cropdims["y"]) + ', w: ' + str(
			model.cropdims["w"]) + ', h: ' + str(model.cropdims["h"])

		if model.comment != '':
			self.ids.comment.opacity = 1
		else:
			self.ids.comment.opacity = 0

		self.ids.ok_sample_header.opacity = 1
		self.ids.photo.opacity = 1
		self.ids.photo.source = model.data_path + '/ok_sample.jpg'
		print("Loaded model:", model.name, model.get_time_created(), model.author, model.comment)

	def clear_model_data(self):
		self.ids.model_name.text = "N/A"
		self.ids.created_date.text = "N/A"
		self.ids.author.text = "N/A"
		self.ids.comment.text = "N/A"
		self.ids.num_iterations.text = "N/A"
		self.ids.batch_size.text = "N/A"
		self.ids.num_ok.text = "N/A"
		self.ids.num_nok.text = "N/A"
		self.ids.threshold.text = "N/A"
		self.ids.sample_name.text = "N/A"
		self.ids.comment.opacity = 0
		self.ids.photo.opacity = 0
		self.ids.ok_sample_header.opacity = 0

	# get names of the model dropdown menu
	def get_values(self):
		values = []
		self.load_list()
		if self.model_list.is_empty():
			values.append("N/A")
		else:
			for item in self.model_list.get_list():
				values.append(item.name)
		return values

	def on_spinner_select(self, name):
		model = self.model_list.find_first(name)
		if model is not None:
			model_name = model.name
			self.model_list.set_selected(model_name)
			self.set_model_data(model_name)

	# show info about selected model
	def show_selected(self):
		if not self.model_list.is_empty():
			self.enable_button(self.ids.import_btn)
			model = self.model_list.get_selected()
			if model is None:  # show last model if none has been selected
				model = self.model_list.get_list()[-1]
			model_name = model.name
			self.set_model_data(model_name)
		else:
			print("No elements")
			self.clear_model_data()
			self.disable_button(self.ids.import_btn)

	def disable_button(self, button):
		button.disabled = True
		button.opacity = .5

	def enable_button(self, button):
		button.disabled = False
		button.opacity = 1

	def open_file_dialog(self):
		root = tk.Tk()
		root.withdraw()
		file_names = filedialog.askopenfilenames(filetypes=[("Image files", ".jpeg .jpg .png .bmp .tiff")])
		print(file_names)
		self.get_root_window().raise_window()  # set focus on window
		if file_names != '':
			# t = Process(target=self.find_anomalies, args=(file_names,))
			# t.start()
			self.find_anomalies(image_paths=file_names)

	def refresh(self):  # update screen
		self.ids.model_name.values = self.get_values()
		self.show_selected()
		self.ids.dir_icon.opacity = .5
		self.ids.num_files.opacity = 0
		self.ids.photo_preview_box.opacity = 0

	def image_load(self, image_paths, json_model_path:str, model_name:str, selected_model):

		shutil.rmtree(self.temp_folder, ignore_errors=True)
		dirname = os.path.dirname(self.temp_folder)
		Path(dirname).mkdir(parents=True, exist_ok=True)

		imgObj = ImageObject()
		imgObj.__class__.size = 0
		IMAGE_DIMS = self.readImageSize(json_model_path=json_model_path, model_name=model_name)
		lstImgObj = []
		imgsArrNpByte = []

		x = selected_model.cropdims['x']
		y = selected_model.cropdims['y']
		w = selected_model.cropdims['w']
		h = selected_model.cropdims['h']

		if not isinstance(x, int):
			x = int(x)
		if not isinstance(y, int):
			y = int(y)
		if not isinstance(w, int):
			w = int(w)
		if not isinstance(h, int):
			h = int(h)

		if w != 0 and h != 0:
			for path in image_paths:
				if path != '':
					if path.endswith(".png") and not path.startswith("._"):
						print("fileAddress: " + path)
						photo_name = os.path.basename(path)
						image = cv2.imread(path)
						cropped = image[y:y + h, x:x + w]
						# write the cropped image to disk in PNG format
						cv2.imwrite(f"{dirname}/{photo_name}", cropped)
						print("Cropped: " + f"{dirname}/{photo_name}")

			fds = os.listdir(self.temp_folder)
			image_paths = []
			for img_file in fds:
				pathImg = f'{self.temp_folder}/{img_file}'
				image_paths.append(pathImg)

			if len(image_paths) == 0:
				return (lstImgObj, imgsArrNpByte)

		for path in image_paths:
			image = cv2.imread(path, 0)
			image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
			imgArrByte = img_to_array(image)

			imgObj = ImageObject(path=path, byteArray=imgArrByte)
			lstImgObj.append(imgObj)


		imgsArrByte = []
		for imgObj in lstImgObj:
			imgsArrByte.append(imgObj.byteArray)
		imgsArrNpByte = np.array(imgsArrByte) / 255.0

		return (lstImgObj, imgsArrNpByte)

	def find_anomalies(self, image_paths: list):

		now0 = datetime.now()
		current_hour = (int(now0.strftime("%H")))

		selected_model = self.model_list.get_selected()
		self.preview_index = 0

		model_name = selected_model.name
		model_path = selected_model.data_path + '/model_file.model'
		json_model_path = selected_model.data_path + '/model_data.json'
		json_images_path = selected_model.data_path + '/images_data.json'


		timeDetectionStart = time.time()

		thresh = selected_model.threshold_manual

		with ThreadPoolExecutor(max_workers=2) as executor:
			futureAutoencoder = executor.submit(self.load_model, model_path=model_path, model_name=model_name)
			futureLoadImages = executor.submit(self.image_load, selected_model=selected_model, image_paths=image_paths, json_model_path=json_model_path, model_name=model_name)

			autoencoder = futureAutoencoder.result()
		try:
			(lstImgObj, imgsArrNpByte) = futureLoadImages.result()
			if(len(lstImgObj) == 0):
				return
		except BaseException:
			print("Image is not correct")
			return

		# reconstruction errors
		# make predictions on our image data and initialize our list of
		decoded = autoencoder.predict(imgsArrNpByte)

		print(f"[INFO] Count of imported images: {len(lstImgObj)}")
		self.loaded_photos = lstImgObj
		# show number of photos loaded
		self.ids.dir_icon.opacity = 0
		self.ids.num_files.text = str(len(lstImgObj)) + ' loaded'
		self.ids.num_files.opacity = 1

		# loop over all original images and their corresponding
		# reconstructions
		errors = []
		for (imgObj, recon) in zip(lstImgObj, decoded):
			# compute the mean squared error between the ground-truth image
			# and the reconstructed image, then add it to our list of errors
			mse = np.mean((imgObj.byteNpArray - recon) ** 2)
			errors.append(mse)
			imgObj.mse = mse

		idxs_nok = np.where(np.array(errors) >= thresh)[0]
		print("[INFO] mse threshold: {}".format(thresh))
		print("[INFO] {} outliers found".format(len(idxs_nok)))

		# loop over the indexes of images with a high mean squared error term

		if (len(idxs_nok) != 0):
			print("[INFO] List of outliers:")
			iter = 1
			for i in idxs_nok:
				print(f'{iter} – {lstImgObj[i].path}')
				iter += 1
				lstImgObj[i].status = False



		timeDetectionStop = time.time() - timeDetectionStart
		print(f"[INFO] time detection anomalies: {timeDetectionStop}")

		countNOK = len(idxs_nok)
		countOK = len(lstImgObj) - countNOK


		self.show_result(self.preview_index)  # show results of first image from the set

		data = [current_hour, timeDetectionStop, 1]
		with open('basic_data', "w", newline='') as csvfile:
			file_wr = csv.writer(csvfile, delimiter=',')
			file_wr.writerow(data)



		thr = Thread(target=self.saveImageDatailsToJSON, args=(json_images_path, lstImgObj))
		thr.start()

	def load_model(self, model_path : str="", model_name : str=""):

		# load the model and image data from disk
		print("[INFO] loading autoencoder and image data...")
		if self.autoencoder == None or self.autoencoderName != model_name:
			try:
				self.autoencoder = load_model(model_path, compile=False)  # True
				self.autoencoderName = model_name

				print(model_name)
				print(model_path)
			except IOError:
				print("Loading model error")

		return self.autoencoder



	def readImageSize(self, json_model_path:str, model_name:str):

		IMAGE_DIMS = []
		with open(json_model_path, "r") as read_file:
			data = json.load(read_file)

		if (data[information_k][model_name_k] == model_name):
			IMAGE_DIMS.append(data[data_k][resized_size_image_w_k])
			IMAGE_DIMS.append(data[data_k][resized_size_image_h_k])
		return IMAGE_DIMS



	def saveImageDatailsToJSON(self, json_images_path, lstImgObj):

		print("[INFO] saving data of tested image to .json...")
		imagesDataJSON = []
		for imgData in lstImgObj:
			data = {
				index_k: int(imgData.index),
				path_k: imgData.path,
				status_k: imgData.status,
				mse_k: float(imgData.mse)
			}
			imagesDataJSON.append(data)

		dataJSON = {
			images_k: imagesDataJSON
		}

		with open(json_images_path, "w") as write_file:
			json.dump(dataJSON, write_file)
			json.dumps(dataJSON, indent=4)

		print("[INFO] Saved to .json\n")


	def show_result(self, index):
		if len(self.loaded_photos) > 0:
			self.ids.photo_preview_box.opacity = 1
			ok_sum = 0
			nok_sum = 0
			for image in self.loaded_photos:
				if image.status == True:
					ok_sum += 1
				else:
					nok_sum += 1

			image = self.loaded_photos[index]
			self.ids.preview_photo.source = image.path
			photo_name = os.path.basename(image.path)
			self.ids.preview_photo_name.text = photo_name + ' (' + str(index + 1) + '/' + str(
				len(self.loaded_photos)) + ')'
			self.ids.photo_threshold.text = 'Threshold: ' + str(image.mse)
			self.ids.preview_ok_number.text = str(ok_sum)
			self.ids.preview_nok_number.text = str(nok_sum)

			if image.status == True:
				self.ids.preview_result.text = "OK"
				self.ids.preview_result.color = (82 / 255, 184 / 255, 111 / 255, 1)
			else:
				self.ids.preview_result.text = "NOK"
				self.ids.preview_result.color = (210 / 255, 48 / 255, 64 / 255, 1)

	def next_image(self):
		if self.preview_index < len(self.loaded_photos) - 1:
			self.preview_index += 1
			self.show_result(self.preview_index)

	def previous_image(self):
		if self.preview_index > 0:
			self.preview_index -= 1
			self.show_result(self.preview_index)
