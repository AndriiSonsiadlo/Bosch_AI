#Copyright (C) 2020 BardzoFajnyZespol

from kivy.uix.screenmanager import Screen
from classes.model_list import ModelList
from classes.my_popup import MyPopup
from functions.gpu_info import *

class LearningMode(Screen):
    model_name = "N/A"
    created = "N/A"
    author = "N/A"
    comment = "N/A"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_list = ModelList()

    def load_list(self):
        self.model_list = ModelList()
        #self.ids.gpu_name.text = get_gpu_name()

    # display model info on screen
    def set_model_data(self, name):
        model = self.model_list.find_first(name)
        self.model_list.set_selected(name)
        self.ids.model_name.text = model.name
        self.ids.created_date.text = str(model.created)
        self.ids.author.text = model.author
        self.ids.comment.text = model.comment
        self.ids.num_iterations.text = str(model.epochs)
        if model.comment != '':
            self.ids.comment.opacity = 1
        else:
            self.ids.comment.opacity = 0
        self.ids.num_ok.text = str(len(model.ok_photos))
        self.ids.num_nok.text = str(len(model.nok_photos))

        if model.threshold_manual == model.threshold:
            self.ids.threshold.text = str("{:.5f}".format(model.threshold)) + '(auto)'
        else:
            self.ids.threshold.text = str("{:.5f}".format(model.threshold_manual)) + '(manual)'
        self.ids.crop_xywh.text = 'x: ' + str(model.cropdims["x"]) + ', y: ' + str(model.cropdims["y"]) + ', w: ' + str(model.cropdims["w"]) + ', h: ' + str(model.cropdims["h"])
        self.ids.batch_size.text = str(model.batch_size)
        self.ids.ok_sample_header.opacity = 1
        self.ids.photo.opacity = 1
        self.ids.photo.source = model.data_path+'/ok_sample.jpg'
        #self.ids.sample_name.text = model.ok_sample #display sample photo filename
        print("Loaded model:", model.name, model.get_time_created(), model.author, model.comment, model.data_path)

    # clear on screen model info
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

    #get names of the model dropdown menu
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
            self.enable_button(self.ids.edit_btn)
            self.enable_button(self.ids.delete_btn)

            model = self.model_list.get_selected()
            if model is None:  # show last model if none has been selected
                model = self.model_list.get_list()[-1]
            model_name = model.name
            self.set_model_data(model_name)
        else:
            print("No elements")
            self.clear_model_data()
            self.disable_button(self.ids.edit_btn)
            self.disable_button(self.ids.delete_btn)

    def disable_button(self, button):
        button.disabled = True
        button.opacity = .5

    def enable_button(self, button):
        button.disabled = False
        button.opacity = 1

    def show_popup(self):
        selected = self.model_list.get_selected()
        if selected is not None:
            popupWindow = MyPopup()
            popupWindow.bind(on_dismiss=self.popup_refresh)
            popupWindow.open()

    def popup_refresh(self, instance): #update screen after pressing delete
        self.load_list()
        self.refresh()

    def refresh(self): #update screen
        self.ids.model_name.values = self.get_values()
        self.show_selected()
