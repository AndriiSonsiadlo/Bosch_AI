#Copyright (C) 2020 BardzoFajnyZespol

import json
import os
from threading import Thread

import numpy as np

import cv2
import tensorflow as tf
from keras_preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from classes._key_json import *
from classes.model import Model
from classes.model_list import ModelList
from kivy.uix.screenmanager import Screen
from classes.mq_subscriber import Subscriber
from kivy.graphics import Color

class AutoMode(Screen):
	json_settins_path = "settings.json"

	server_is_started = False
	server_is_connecting = False
	isBrokenSub = False
	subscriber = None
	config = {}
	none_ip_server = "Disconnected"
	connecting_status = "Connecting..."
	loading_status = "Loading models..."

	def __init__(self, **kw):
		super().__init__(**kw)
		self.update_model_list()
		self.start_fun()

	def __del__(self):
		self.stop_fun(close=True)
		try:
			self.subscriber.close_connection()
			del self.subscriber
			self.subscriber = None
			exit(0)
		except BaseException:
			pass

	def set_server_ip(self, ip):
		self.ids.server_ip.text = ip
		if ip != self.none_ip_server and ip != self.connecting_status and ip != self.loading_status:
			self.ids.connection_status.col = (82/255,184/255,111/255,1)
		else:
			self.ids.connection_status.col = (.7, .7, .7, 1)

	def test(self, test):
		self.ids.server_ip.text = test
		self.ids.part_number.text = 'test'
		self.ids.learning_model.text = 'test'

	def start_fun(self):

		if (not self.get_config()):
			return

		try:
			self.server_is_started = self.subscriber.isConnection
			self.server_is_connecting = self.subscriber.isConnecting
			self.isBrokenSub = self.subscriber.isBroken
		except BaseException:
			self.server_is_started = False
			self.server_is_connecting = False

		if ((self.server_is_started == False and self.server_is_connecting == False) or self.isBrokenSub == True):
			try:
				if(self.config[photo_path_conf_k] != ""):
					self.subscriber = Subscriber(self.config, automode_ref=self)
					self.server_is_connecting = True
					self.subscriber.start()
				else:
					print(" [*] The path to photos is empty")

			except BaseException:
				self.subscriber = None

	def stop_fun(self, close=False):
		try:
			self.server_is_started = self.subscriber.isConnection
			self.server_is_connecting = self.subscriber.isConnecting
		except BaseException:
			self.server_is_started = False
			self.server_is_connecting = False

		if (self.server_is_started == True or self.server_is_connecting == True or close == True):
			try:
				self.set_server_ip(self.none_ip_server)

				self.subscriber.close_connection()
			except:
				print("Connection force stopped")
				self.set_server_ip(self.none_ip_server)

			self.server_is_started = False
			self.subscriber = None

	def get_config(self):

		try:
			with open(self.json_settins_path, "r") as read_file:
				data = json.load(read_file)

			ip = data[server_ip_k]
			port = data[port_number_k]
			password = data[password_k]
			username = data[username_k]
			pub_key = data[publisher_key_k]
			sub_key = data[subscriber_key_k]
			timeout = data[timeout_k]
			path = data[photo_path_k]

			self.config = {username_conf_k: username, password_conf_k: password,
			               host_conf_k: ip, port_conf_k: port,
			               sub_key_conf_k: sub_key, pub_key_conf_k: pub_key,
			               timeout_conf_k: timeout, photo_path_conf_k: path}
			return True
		except BaseException:
			print(f" [*] File {self.json_settins_path} is not exist")




	def update_model_list(self):  # check directory names in 'model data' directory update the model list
		model_data_dir = 'model data'
		if (os.path.isdir(model_data_dir)):
			model_list = ModelList()
			model_list.clear_list()
			for file in os.listdir(model_data_dir):
				if os.path.isdir(os.path.join(model_data_dir, file)):
					model_name = file
					print('name', model_name)
					try:
						with open(model_data_dir + '/' + model_name + '/model_data.json', "r") as read_file:
							model_data = json.load(read_file)
							print(model_data)

							new_model = Model(model_name, model_data[information_k][author_k],
							                  model_data[information_k][comment_k])
							new_model.created = model_data[information_k][d_m_y_k] + ' ' + model_data[information_k][
								h_m_s_k]
							new_model.ok_photos = model_data[information_k][ok_photos_k]
							new_model.nok_photos = model_data[information_k][nok_photos_k]
							new_model.automode_selected = model_data[information_k][automode_k]
							new_model.data_path = model_data_dir + '/' + model_name
							new_model.epochs = int(model_data[data_k][epochs_k])
							new_model.batch_size = int(model_data[data_k][batch_size_k])
							new_model.learning_time = float(model_data[information_k][learning_time_k])
							new_model.threshold = float(model_data[data_k][thresh_auto_k])
							new_model.threshold_manual = float(model_data[data_k][thresh_manual_k])
							new_model.cropdims = model_data[data_k][crop_dimenstion_k]
							model_list.add_model(new_model)
					except IOError:
						print('Model name:', model_name, 'error. No JSON file')

	def show_photo(self, source: str):
		print(source)
		self.ids.photo.source = source

	def show_photo_name(self, name: str):
		self.ids.photo_name.text = os.path.basename(name)

	def show_result(self, result: bool):
		if result:
			self.ids.check_result.text = 'OK'
			self.ids.check_result.color = (82 / 255, 184 / 255, 111 / 255, 1)
		else:
			self.ids.check_result.text = 'NOK'
			self.ids.check_result.color = (210 / 255, 48 / 255, 64 / 255, 1)
