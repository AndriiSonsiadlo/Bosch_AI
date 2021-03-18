#Copyright (C) 2020 BardzoFajnyZespol

# Crops all images in given directory into w by h images, starting at x,y.
# (x=0,y=0) is left upper corner. Supports only PNG
# Result images are written to new "/cropped" directory and named 1.png, 2.png [...]
# There is no input validation

import os
from pathlib import Path
import cv2
import os.path

def crop(x,y,w,h,image_path = ""):
    dirname = os.path.dirname(image_path)
    dirname = os.path.join(dirname, 'cropped')

    filename = os.path.basename(image_path)

    Path(dirname).mkdir(parents=True, exist_ok=True)  # /cropped directory is created in given path
    if not isinstance(x, int):
        x = int(x)
    if not isinstance(y, int):
        y = int(y)
    if not isinstance(w, int):
        w = int(w)
    if not isinstance(h, int):
        h = int(h)

    if image_path != '' and w != 0 and h != 0:
        if image_path.endswith(".png") and not image_path.startswith("._"): # bulletproofing against thumbnail files
            OutName = filename
            #if directory[-1:] != '/':
                #directory = directory + '/'
            print("fileAddress: " + image_path)
            image = cv2.imread(image_path)
            cropped = image[y:y+h, x:x+w]

            # write the cropped image to disk in PNG format
            cv2.imwrite(f"{dirname}/{OutName}", cropped)
            print("Cropped: " + f"{dirname}/{OutName}")

        cv2.destroyAllWindows()
    # Example usage:
    # dir="/Volumes/Karcioszka/Bosch/Work_copy/OK_copy/"
    #dir = "/Volumes/Karcioszka/Bosch/Work_copy/OK_copy/"
    #crop(1140,940,350,200,dir)
    #crop(1582,980,50,120,dir)
