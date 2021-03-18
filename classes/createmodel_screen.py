#Copyright (C) 2020 BardzoFajnyZespol


import os

from kivy.clock import mainthread
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.screenmanager import Screen
from classes.model_list import ModelList
from classes.model import Model
from classes.plot_popup import PlotPopup
import tkinter as tk
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functions.gpu_info import *
from multiprocessing import Process

class LearningCreate(Screen):
    author = ""
    comment = ""
    train_btn_enabled_text = 'Train model'
    header_text_color = (0 / 255, 102 / 255, 178 / 255, 1)
    normal_text_color = (0.2, 0.2, 0.2, 1)
    isLearning = False
    new_model = Model()


    def __init__(self, **kw):
        super().__init__(**kw)
        #self.new_model = Model()

    def get_gpu_name(self):
        self.ids.gpu_name.text = get_gpu_name()

    @mainthread
    def clear_inputs(self):

        if (self.isLearning):
            self.enable_learning_btn()
            self.ids.begin_learning_button.color = self.normal_text_color
            self.ids.begin_learning_button.text = "Learning..."
        else:
            self.ids.create_model_name.text = ''
            self.ids.create_author.text = ''
            self.ids.create_comment.text = ''
            self.ids.crop_x_input.text = ''
            self.ids.crop_y_input.text = ''
            self.ids.crop_w_input.text = ''
            self.ids.crop_h_input.text = ''
            self.new_model = Model()
            self.ids.learning_results.opacity = 0
            self.ids.cropped_area.source = ''


    def change_text(self):
        if self.ids.begin_learning_button.text == self.train_btn_enabled_text and self.learning_parameters_set():
            self.ids.begin_learning_button.color = self.normal_text_color
            self.ids.begin_learning_button.text = "Learning..."

    def enable_learning_btn(self):  # enables training button
        self.ids.begin_learning_button.disabled = False
        self.ids.begin_learning_button.text = self.train_btn_enabled_text
        self.ids.begin_learning_button.opacity = 1
        self.ids.begin_learning_button.color = self.header_text_color
        self.ids.learning_results.opacity = 0


    def begin_learning(self):

        if(self.isLearning == False):
            self.isLearning = True
            t = threading.Thread(target=self.begin_learning_release, args=("", ""), daemon=True)
            t.start()




    def begin_learning_release(self, str1, str2):
        if self.ids.begin_learning_button.text == "Learning...":
            self.new_model.name = self.ids.create_model_name.text
            self.new_model.author = self.ids.create_author.text
            self.new_model.comment = self.ids.create_comment.text

            model_list = ModelList()

            if self.new_model.name == "":
                self.new_model.name = "unnamed"
            if self.new_model.author == "":
                self.new_model.author = "Unknown"

            if model_list.check_name_exists(self.new_model.name):
                print("File", self.new_model.name, "already exists")
                repeated = 1
                while model_list.check_name_exists(self.new_model.name + "(" + str(repeated) + ")"):
                    repeated += 1
                self.new_model.name += "(" + str(repeated) + ")"


            data_dir = self.new_model.data_path
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)

            self.new_model.data_path += '/' + self.new_model.name

            if not os.path.isdir(self.new_model.data_path):
                os.mkdir(self.new_model.data_path)


            if self.ids.dimensions_box.opacity == 1:
                self.new_model.cropdims["x"] = int(self.ids.crop_x_input.text)
                self.new_model.cropdims["y"] = int(self.ids.crop_y_input.text)
                self.new_model.cropdims["w"] = int(self.ids.crop_w_input.text)
                self.new_model.cropdims["h"] = int(self.ids.crop_h_input.text)

            if self.ids.epoch_checkbox.active == False:
                early_stopping = True
            else:
                early_stopping = False

            if early_stopping == True:
                iternum = 1000
            else:
                iternum = int(self.ids.create_iter_num.text)
            bs = int(self.ids.create_batch_size.text)

            self.new_model.begin_learning(epochs=iternum, batch_size=bs, early_stopping=early_stopping)
            print(self.new_model.learning_time)
            try:
                self.new_model.save_ok_sample(
                self.new_model.ok_photos[0])  # saving ok photo sample (temporary solution - takes first photo from the dataset)
            except BaseException:
                self.isLearning = False

            model_list.add_model(self.new_model)
            model_list.set_selected(model_list.get_list()[-1].name)
            self.ids.begin_learning_button.text = "Completed"

            self.show_results(learning_time=self.new_model.learning_time, threshold=self.new_model.threshold)

            self.show_plot(data_path=self.new_model.data_path)

            self.ids.begin_learning_button.disabled = True
            self.ids.begin_learning_button.opacity = .5
            self.new_model = Model()
            self.isLearning = False




    def learning_parameters_set(self):  # checks if learning parameters are set and are numbers
        if self.ids.crop_checkbox.active:
            return (self.ids.create_iter_num.text != '' or self.ids.create_iter_num.hint_text == 'auto') and self.ids.create_batch_size.text != '' and \
                   self.ids.crop_x_input.text != '' and self.ids.crop_x_input.text.isnumeric() and \
                   self.ids.crop_y_input.text != '' and self.ids.crop_y_input.text.isnumeric() and \
                   self.ids.crop_w_input.text != '' and self.ids.crop_w_input.text.isnumeric() and \
                   self.ids.crop_h_input.text != '' and self.ids.crop_h_input.text.isnumeric()
        else:
            return (self.ids.create_iter_num.text != '' or self.ids.create_iter_num.hint_text == 'auto') and self.ids.create_batch_size.text != ''

    @mainthread
    def show_results(self, learning_time, threshold):
        self.ids.learning_results.text = "Learning time: " + str("{:.2f}".format(learning_time)) +\
                                         ' s, MSE threshold: '+str("{:.7f}".format(threshold))
        self.ids.learning_results.opacity = 1


    def save_model(self):
        if self.ids.begin_learning_button.text == 'OK':
            self.ids.begin_learning_button.text = self.train_btn_enabled_text
            self.manager.current = "learning"

    def cropdims_to_model(self):
        self.new_model.cropdims["x"] = self.ids.crop_x_input.text if self.ids.crop_x_input.text != '' else 0
        self.new_model.cropdims["y"] = self.ids.crop_y_input.text if self.ids.crop_y_input.text != '' else 0
        self.new_model.cropdims["w"] = self.ids.crop_w_input.text if self.ids.crop_w_input.text != '' else 0
        self.new_model.cropdims["h"] = self.ids.crop_h_input.text if self.ids.crop_h_input.text != '' else 0

    def load_nok(self):
        #thread = threading.Thread(target=self.load_nok_release())
        #thread.start()
        self.load_nok_release()

    def load_nok_release(self):
        self.cropdims_to_model()
        num_ok_loaded = self.new_model.ok_photos
        num_nok_loaded = self.new_model.get_photos(self, ok=False)
        if num_nok_loaded > 0:
            self.ids.num_nok_files.text = str(num_nok_loaded) + " loaded"
            self.ids.num_nok_files.opacity = 1
            self.ids.nok_dir_icon.opacity = 0
            if len(num_ok_loaded) > 1:
                self.enable_learning_btn()
        self.get_root_window().raise_window()  # set focus on window

    def load_ok(self):
        #thread = threading.Thread(target=self.load_ok_release())
        #thread.start()
        self.load_ok_release()

    def load_ok_release(self):
        self.cropdims_to_model()
        num_nok_loaded = self.new_model.nok_photos
        num_ok_loaded = self.new_model.get_photos(self, ok=True)
        if num_ok_loaded > 0:
            self.ids.num_ok_files.text = str(num_ok_loaded) + " loaded"
            self.ids.num_ok_files.opacity = 1
            self.ids.ok_dir_icon.opacity = 0
            if len(num_nok_loaded) > 1:
                self.enable_learning_btn()
        self.get_root_window().raise_window()  # set focus on window

    def show_crop_boxes(self, value):
        if value is True:
            self.ids.dimensions_box.opacity = 1
            self.ids.checkbox_title.opacity = 0
        else:
            self.ids.dimensions_box.opacity = 0
            self.ids.checkbox_title.opacity = 1
            self.ids.crop_x_input.text = ''
            self.ids.crop_y_input.text = ''
            self.ids.crop_w_input.text = ''
            self.ids.crop_h_input.text = ''
            self.ids.cropped_area.source = ''

    @mainthread
    def show_plot(self, data_path = ""):
        plot_path = os.path.join(data_path, 'plot.png')
        popupWindow = PlotPopup(plot_path)
        popupWindow.open()

    def enable_input(self, value):
        if value is True:
            self.ids.create_iter_num.disabled = False
            self.ids.create_iter_num.text = ''
            self.ids.create_iter_num.hint_text = '0'
        else:
            self.ids.create_iter_num.disabled = True
            self.ids.create_iter_num.text = 'auto'

    def cancel_learning(self):
        print("learning canceled by user")