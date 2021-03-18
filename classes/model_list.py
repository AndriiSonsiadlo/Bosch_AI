#Copyright (C) 2020 BardzoFajnyZespol

import json
import os
import pickle

from classes._key_json import *
from classes.model import Model

class ModelList:
    path = 'model data/model_list.pkl'

    def __init__(self):
        self.list = self.read_from_file()

    def get_list(self):
        return self.list

    def save_to_file(self):
        with open(self.path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def add_model(self, model):

        #this section is commented out because checking name repetition is checked before model learning (in create mode)
        #model_name = model.name
        #author = model.author
        #comment = model.comment

        #if model_name == "":
        #    model_name = "unnamed"
        #if author == "":
        #    author = "Unknown"
        #if comment == "":
        #    comment = "None"


        #if self.check_name_exists(model_name):
        #    print("File", model_name, "already exists")
        #    repeated = 1
        #    while self.check_name_exists(model_name+"("+str(repeated)+")"):
        #        repeated += 1
        #    model_name += "("+str(repeated)+")"

        #model.name = model_name
        #model.author = author
        #model.comment = comment



        #new_model = Model(model_name, author, comment)
        self.list.append(model)
        print("Added:", model.name)
        self.save_to_file()

    def read_from_file(self):
        model_list = []
        try:
            with open(self.path, 'rb') as input:
                model_list = pickle.load(input).get_list()
        except IOError:
            print("File not accessible")
        return model_list

    def get_selected(self):  # returns selected model if exists
        selected = None
        for model in self.list:
            if model.selected:
                selected = model
        return selected

    def set_selected(self, name):  #sets model as selected
        for model in self.list:
            model.selected = False
        self.find_first(name).selected = True
        self.save_to_file()

    def is_empty(self):
        return len(self.list) == 0

    def find_first(self, name):  # finds first model that matches name
        found = None
        for model in self.list:
            if model.name == name:
                found = model
        return found

    def check_name_exists(self, name):
        exists = False
        for m in self.list:
            if m.name == name:
                exists = True
        return exists

    def clear_list(self):
        self.list.clear()
        self.save_to_file()

    def print_list(self):
        for m in self.list:
            print(m.name, m.get_time_created(), m.author, m.comment)

    def delete_model(self, name):
        data_path = self.find_first(name).data_path
        print(data_path)
        try:
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                os.remove(file_path)
            os.rmdir(data_path)
        except Exception :
            print('Failed to delete files')

        name = self.find_first(name).name
        self.list.remove(self.find_first(name))
        self.save_to_file()
        print("Removed:", name)


    def edit_model_name(self, name: str, new_name: str):
        if (self.find_first(name) is not None):
            if self.check_name_exists(new_name):
                print("Name already exists")
            else:
                json_path = self.find_first(name).data_path + '/model_data.json'
                model_data_dir = self.find_first(name).data_path
                name_index = model_data_dir.rfind(name)
                base_dir = model_data_dir[:name_index]

                self.find_first(name).name = new_name
                with open(json_path, "r") as f:
                    data = json.load(f)

                data[information_k][model_name_k] = new_name

                with open(json_path, 'w') as f:
                    json.dump(data, f)
                    json.dumps(data, indent=4)
                print('RENAME from',model_data_dir, 'TO',base_dir + new_name)
                os.rename(model_data_dir, base_dir + new_name)
                self.find_first(new_name).data_path = base_dir + new_name
                self.save_to_file()

    def edit_model_threshold(self, name: str, new_threshold: float):
        if (self.find_first(name) is not None):
            json_path = self.find_first(name).data_path + '/model_data.json'
            self.find_first(name).threshold_manual = new_threshold

            with open(json_path, "r") as f:
                 data = json.load(f)

            data[data_k][thresh_manual_k] = new_threshold

            with open(json_path, 'w') as f:
                json.dump(data, f)
                json.dumps(data, indent=4)

            self.save_to_file()

    def edit_model_description(self, name: str, new_desc: str):
        if (self.find_first(name) is not None):
            json_path = self.find_first(name).data_path + '/model_data.json'
            model_data_dir = self.find_first(name).data_path

            self.find_first(name).comment = new_desc
            with open(json_path, "r") as f:
                data = json.load(f)

            data[information_k][comment_k] = new_desc

            with open(json_path, 'w') as f:
                json.dump(data, f)
                json.dumps(data, indent=4)

            self.save_to_file()

    def edit_model_automode(self, name: str, use: bool):
        if (self.find_first(name) is not None):
            json_path = self.find_first(name).data_path + '/model_data.json'
            model_data_dir = self.find_first(name).data_path

            self.find_first(name).automode_selected = use
            with open(json_path, "r") as f:
                data = json.load(f)

            data[information_k][automode_k] = use

            with open(json_path, 'w') as f:
                json.dump(data, f)
                json.dumps(data, indent=4)

            self.save_to_file()