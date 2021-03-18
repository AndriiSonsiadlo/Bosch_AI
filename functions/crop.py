#Copyright (C) 2020 BardzoFajnyZespol

# Crops all images in given directory into w by h images, starting at x,y.
# (x=0,y=0) is left upper corner. Supports only PNG
# Result images are written to new "/cropped" directory and named 1.png, 2.png [...]
# There is no input validation

import os
from pathlib import Path
import cv2

def crop(x,y,w,h,directory = "./"):
    dirname = f'{directory}/cropped'
    Path(dirname).mkdir(parents=True, exist_ok=True)  # /cropped directory is created in given path
    i=0

    for filename in os.listdir(directory): # iterates through all files in given directory
        if filename.endswith(".png") and not filename.startswith("._"): # bulletproofing against thumbnail files
            i=i+1
            OutName = str(i)
            fileAddress = f"{directory}{filename}"
            image = cv2.imread(fileAddress)
            cropped = image[y:y+h, x:x+w]

            # uncomment following lines for live preview:
            cv2.imshow("Cropped", cropped)
            cv2.waitKey(1)


            # write the cropped image to disk in PNG format
            cv2.imwrite(f"{directory}/cropped/{OutName}.png", cropped)
            print("Cropped: " + fileAddress)

#Example usage:
dir="/Volumes/Karcioszka/Bosch/Work_copy/OK_copy/"
crop(1140,940,350,200,dir)
