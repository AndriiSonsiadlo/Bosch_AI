#Copyright (C) 2020 BardzoFajnyZespol

from classes.widget_styles import RoundButton
from kivy.core.window import Window
from pathlib import Path

#button to drag and drop files on
class DropButton(RoundButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_dropfile=self.on_file_drop)
        self.path = ''


    def on_file_drop(self, window, file_path):
        path = file_path.decode("utf-8")
        path = Path(path)
        current_screen = self.parent.parent.parent.parent
        current_screen_manager = self.parent.parent.parent.parent.manager.current

        button_text = self.children[2].text

        #print('box position', str(self.pos))
        #print('box size', str(self.size))
        #print('mouse position', window.mouse_pos)
        within_box_width = window.mouse_pos[0]>=self.pos[0] and window.mouse_pos[0] <= self.pos[0] + self.size[0]
        within_box_height = window.mouse_pos[1]>=self.pos[1] and window.mouse_pos[1] <= self.pos[1] + self.size[1]

        if within_box_width and within_box_height and current_screen_manager == 'learning_create':
            self.path = path
            print(self.path)
            if 'OK' in button_text:
                current_screen.new_model.get_photos_from_drop(current_screen, ok=True)
            if 'NOK' in button_text:
                current_screen.new_model.get_photos_from_drop(current_screen, ok=False)