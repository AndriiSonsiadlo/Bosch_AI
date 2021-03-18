#Copyright (C) 2020 BardzoFajnyZespol

from kivy.uix.modalview import ModalView
from kivy.uix.behaviors import ButtonBehavior

#delete model popup window
class PlotPopup(ModalView):
    def __init__(self, plot_path, **kwargs):
        super().__init__(**kwargs)
        self.ids.plot.source = plot_path

    def close(self):
        self.dismiss()
