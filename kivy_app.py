#Copyright (C) 2020 BardzoFajnyZespol

import threading

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
#importing classes. do not remove
from classes.model import Model
from classes.model_list import ModelList
from classes.automode_screen import AutoMode
from classes.learningmode_screen import LearningMode
from classes.my_popup import MyPopup
from classes.editmodel_screen import LearningEdit
from classes.createmodel_screen import LearningCreate
from classes.statistics_screen import Statistics
from classes.manualmode_screen import ManualMode
from classes.settings_screen import SettingsScreen
from classes.settings_screen import SettingsScreen
from classes.drop_button import DropButton
from classes.screen_stack import ScreenStack

#loading ui files
Builder.load_file("ui files/widget_styles.kv")
Builder.load_file("ui files/boschai_ui.kv")
Builder.load_file("ui files/automode_screen.kv")
Builder.load_file("ui files/learningmode_screen.kv")
Builder.load_file("ui files/editmodel_screen.kv")
Builder.load_file("ui files/createmodel_screen.kv")
Builder.load_file("ui files/manualmode_screen.kv")
Builder.load_file("ui files/statistics_screen.kv")
Builder.load_file("ui files/settings_screen.kv")
Builder.load_file("ui files/my_popup.kv")
Builder.load_file("ui files/plot_popup.kv")


#Main Screen with navigation bar on top
class Main(GridLayout, threading.Thread):
    manager = ObjectProperty(None)



#manager for changing screens
class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stack = ScreenStack()
        self.stack.add_screen("auto")

#main app class
class BoschAIApp(App):
    icon = 'Images/icon.png'

    Window.minimum_width, Window.minimum_height = (800,600)

    def build(self):
        # showing main screen
        return Main()

if __name__ == '__main__':
    BoschAIApp().run()
