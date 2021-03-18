#Copyright (C) 2020 BardzoFajnyZespol

from kivy.uix.popup import Popup
from classes.model_list import ModelList

#delete model popup window
class MyPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.list = ModelList()
        self.selected = self.list.get_selected()
        self.title = "Are you sure you want to delete '" + self.selected.name + "'?"

    def yes_pressed(self):
        print("yes")
        self.list.delete_model(self.selected.name)
        self.dismiss()

    def no_pressed(self):
        print("no")
        self.dismiss()