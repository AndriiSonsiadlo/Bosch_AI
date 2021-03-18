#Copyright (C) 2020 BardzoFajnyZespol

import os
import sys
from tkinter import filedialog
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
import json
import time
from classes._key_json import *


class SettingsScreen(Screen):

    def enable_save(self):
        self.ids.save_button.text = 'Apply settings'
        self.ids.save_button.disabled = False
        self.ids.save_button.opacity = 1

    def disable_save(self):
        self.ids.save_button.text = 'Saved'
        self.ids.save_button.disabled = True
        self.ids.save_button.opacity = .5

    def save_settings_to_file(self):
        settings = {}
        connection = self.ids.connection_settings
        key = ''
        for child in connection.children:
                for grandchild in child.children:
                    if str(type(grandchild)) == "<class 'kivy.uix.label.Label'>":
                        key = grandchild.text.lower().replace(': ',"")
                    if str(type(grandchild)) == "<class 'kivy.factory.CustomInput'>":
                        value = grandchild.text
                if key != '':
                    settings[key] = value

        other = self.ids.other_settings
        for child in other.children:
                for grandchild in child.children:
                    if str(type(grandchild)) == "<class 'kivy.uix.label.Label'>":
                        key = grandchild.text.lower().replace(': ',"")
                    if str(type(grandchild)) == "<class 'kivy.factory.CustomInput'>":
                        value = grandchild.text
                if key != '':
                    settings[key] = value

        self.create_json(settings)
        self.disable_save()

    def load_settings_to_inputs(self, file_path):
        self.enable_save()

        data = []
        if os.path.isfile(file_path):
            with open(file_path, "r") as read_file:
                print("[INFO] Loading settings from JSON")
                data = json.load(read_file)
                self.ids.sub_key.text = data[subscriber_key_k]
                self.ids.pub_key.text = data[publisher_key_k]
                self.ids.username.text = data[username_k]
                self.ids.password.text = data[password_k]
                self.ids.port.text = data[port_number_k]
                self.ids.server_ip.text = data[server_ip_k]
                self.ids.timeout.text = data[timeout_k]
                self.ids.path.text = data[photo_path_k]

    def create_json(self, settings_dict):
        file_path = 'settings.json'
        print("[INFO] Saving settings to JSON")
        with open(file_path, "w") as write_file:
            json.dump(settings_dict, write_file)
            json.dumps(settings_dict, indent=4)

    def open_dir_dialog(self):
        path = filedialog.askdirectory()
        if path != '':
            self.ids.path.text = path
        self.get_root_window().raise_window()