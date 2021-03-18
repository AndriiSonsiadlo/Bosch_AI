#Copyright (C) 2020 BardzoFajnyZespol

import threading
import time
from threading import Thread

import cv2
import pika
import paho.mqtt.client as mqtt
from keras_preprocessing.image import img_to_array

from classes._key_json import *
from classes.automode_find_anomalies import Anomalies
from classes.model_list import ModelList
from tensorflow.keras.models import load_model
import csv
from datetime import datetime
import numpy as np

class Subscriber(Thread):
	def __init__(self, config, automode_ref):
		Thread.__init__(self)
		self.stop_event = threading.Event()

		self.automode_ref = automode_ref
		self.client: mqtt.Client = None

		self.isConnection = False
		self.isConnecting = False
		self.isBroken = False


		self.broker = config[host_conf_k]
		self.broker_port = int(config[port_conf_k])
		self.timeout_reconnect = int(config[timeout_conf_k])
		self.broker_topic_sub = config[sub_key_conf_k]  # image topic
		self.broker_topic_pub = config[pub_key_conf_k]  # result topic
		self.username = config[username_conf_k]
		self.password = config[password_conf_k]
		self.photo_path = config[photo_path_conf_k]

		self.isLog = False

		self.config = config

		self.autoencoders = {}
		self.autoencoders_info = {}
		self.model_list = None

	def __del__(self):

		self.isConnection = False
		self.client = None
		self.isBroken = True
		self.stop_event.set()

	#		print(" [*] Connection have been deleted\n")

	def close_connection(self):
		try:
			self.automode_ref.set_server_ip(self.automode_ref.none_ip_server)
			self.isConnecting = False
			self.isConnection = False

			self.isBroken = True
			self.client.disconnect()
			print(" [*] Connection have been closed\n")
		except BaseException:
			print(" [*] Error to close a connection")

		self.stop_event.set()

	def run(self):
		try:
			print(" [*] Connection is creating")

			self.isConnecting = True

			while True:
				try:
					self.automode_ref.set_server_ip(self.automode_ref.connecting_status)
					break
				except:
					pass

			self._create_connection()

			self.choose_models()


			if(len(self.autoencoders_info) != 0):
				print(f" [*] Loading models...")
				self.automode_ref.set_server_ip(self.automode_ref.loading_status)
				self.load_model()
				self.automode_ref.set_server_ip(self.config[host_conf_k])
				self.isConnecting = False
				self.isConnection = True
				print(f" [*] Loaded {len(self.autoencoders_info)} models")
			else:
				print("Not found model for automode")
				self.close_connection()
				return

			self.setup()
		except BaseException:
			print(" [*] Connection is force closed")
			while True:
				try:
					self.automode_ref.set_server_ip(self.automode_ref.none_ip_server)
					break
				except:
					pass
			self.isConnection = False
			self.isConnecting = False
			self.client = None
			self.isBroken = True

	# A creation connection to a server
	def _create_connection(self):

		self.client = mqtt.Client()

		self.client.on_connect = self.on_connect
		self.client.on_message = self.on_message
		self.client.on_publish = self.on_publish
		self.client.on_subscribe = self.on_subscribe
		self.client.on_log = self.on_log

		self.client.connect(host=self.broker, port=self.broker_port, keepalive=self.timeout_reconnect)
		self.client._client_id = "AppAI_PWr"
		self.client.username_pw_set(self.username, self.password)

	def setup(self):
		try:
			print(" [*] Connection have been created ")
			self.isConnection = True

			self.client.loop_forever()

			self.isConnection = False

		except BaseException:
			self.isConnection = False

	def on_connect(self, client, userdata, flags, rc):
		# value of rc determines success or not

		if int(str(rc)) == 0:
			print(" [LOG] on_connect: client connection successful")
			self.client.subscribe(self.broker_topic_sub, qos=0)
		elif int(str(rc)) == 1:
			print(" [LOG] on_connect: client connection refused - incorrect protocol version")
		elif int(str(rc)) == 2:
			print(" [LOG] on_connect: client connection refused - invalid client identifier")
		elif int(str(rc)) == 3:
			print(" [LOG] on_connect: client connection refused - server unavailable")
		elif int(str(rc)) == 4:
			print(" [LOG] on_connect: client connection refused - bad username or password")
		elif int(str(rc)) == 5:
			print(" [LOG] on_connect: client connection refused - not authorised")

	def on_message(self, client, userdata, msg):

		image_name = str(msg.payload.decode("utf-8"))
		print(f"Received in topic: {str(msg.topic)}; message: {image_name}")

		if image_name.startswith("CheckResult"):

			now0 = datetime.now()
			current_hour = (int(now0.strftime("%H")))

			time_start = time.time()

			checkAnomaly = Anomalies(image_name, autoencoders_info=self.autoencoders_info, autoencoders=self.autoencoders, automode_ref=self.automode_ref, photo_path=self.photo_path)
			checkAnomaly.start()
			checkAnomaly.join()

			if  checkAnomaly.status != None:

				time_stop = time.time() - time_start

				print(f"Published Result: {checkAnomaly.status}")
				if  checkAnomaly.status == True:
					self.client.publish(self.broker_topic_pub, f"Result{checkAnomaly.image_name}_OK")
				else:
					self.client.publish(self.broker_topic_pub, f"Result{checkAnomaly.image_name}_NOK")

			data = [current_hour, time_stop, checkAnomaly.status]
			with open('basic_data', "a", newline='') as csvfile:
				file_wr = csv.writer(csvfile, delimiter=',')
				file_wr.writerow(data)


	def on_publish(self, mosq, obj, mid):
		#print("mid: " + str(mid))
		pass

	def on_subscribe(self, mosq, obj, mid, granted_qos):
		print(" [LOG] on_subscribe: " + str(mid) + " " + str(granted_qos))

	def on_log(self, mosq, obj, mid, string):
		if (self.isLog == True):
			print(string)
		pass


	def choose_models(self):

		model_list_temp = ModelList()
		self.model_list = model_list_temp.get_list()

		temp_autoencoders_info = {}

		for model in self.model_list:
			if model.automode_selected == True:
				temp_autoencoders_info[model.name] = model

		self.autoencoders_info = temp_autoencoders_info

	def load_model(self):
		for model in self.autoencoders_info:
			model_path = self.autoencoders_info[model].data_path + '/model_file.model'
			try:
				temp_autoencoder = load_model(model_path, compile=False)  # True
				self.autoencoders[model] = temp_autoencoder
			except IOError:
				print("Loading model error")

		self.preload_tf()



	def preload_tf(self):

		for model in self.autoencoders_info:
			model_path = f"model data/{self.autoencoders_info[model].name}/model_file.model"
			image_path = f"model data/{self.autoencoders_info[model].name}/ok_sample.jpg"
			try:
				temp_autoencoder = load_model(model_path, compile=False)  # True
			except IOError:
				print("Preloading model error")
				return

			image = cv2.imread(image_path, 0)
			image = cv2.resize(image, (200, 200))

			imgsArrByte = []
			imgsArrByte.append(img_to_array(image))
			imgsArrNpByte = np.array(imgsArrByte) / 255.0

			decoded = temp_autoencoder.predict(imgsArrNpByte)
			break
