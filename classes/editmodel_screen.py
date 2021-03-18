#Copyright (C) 2020 BardzoFajnyZespol

from kivy.uix.screenmanager import Screen
from classes.model_list import ModelList

class LearningEdit(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_list = ModelList()
        self.description_deleted = False

    def load_list(self):
        self.model_list = ModelList()

    #display model info on screen
    def set_model_data(self, list, name):
        model = list.find_first(name)
        self.ids.model_name.hint_text = model.name
        self.ids.created_date.text = model.get_time_created()
        self.ids.author.text = model.author
        if model.comment != '':
            self.ids.description.hint_text = model.comment
        else:
            self.ids.description.hint_text = "No description"
        self.ids.threshold.text = str("{:.5f}".format(model.threshold))
        self.ids.automode_checkbox.active = model.automode_selected


    #show info about selected model
    def show_selected(self):
        if not self.model_list.is_empty():
            model = self.model_list.get_selected()
            model_name = model.name
            self.set_model_data(self.model_list, model_name)

    def save_edited_model(self):
        model = self.model_list.get_selected()
        if (self.ids.model_name.text != ''):
            self.model_list.edit_model_name(model.name, self.ids.model_name.text)
        if (self.ids.description.text != '' or self.ids.description.hint_text != model.comment):
            self.model_list.edit_model_description(model.name, self.ids.description.text)
        if (self.description_deleted):
            self.model_list.edit_model_description(model.name, '')
        if (self.ids.threshold_manual.text != ''):
            self.model_list.edit_model_threshold(model.name, float(self.ids.threshold_manual.text))
        else:
            self.model_list.edit_model_threshold(model.name, model.threshold)
        self.model_list.edit_model_automode(model.name, self.ids.automode_checkbox.active)


    def clear_inputs(self):
        self.description_deleted = False
        self.ids.model_name.text = ''
        self.ids.author.text = ''
        self.ids.description.text = ''
        model = self.model_list.get_selected()
        if model.threshold_manual != model.threshold:
            self.ids.manual_checkbox.active = True
            self.ids.threshold_manual.text = str("{:.5f}".format(model.threshold_manual))

    def enable_input(self, value):
        if value is True:
            self.ids.threshold_manual.disabled = False
            self.ids.threshold_manual.text = '0'
        else:
            self.ids.threshold_manual.disabled = True
            self.ids.threshold_manual.text = ''

    def delete_description(self):
        model = self.model_list.get_selected()
        self.ids.description.text = ''
        self.ids.description.hint_text = 'description deleted'
        self.description_deleted = True
