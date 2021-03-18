#Copyright (C) 2020 BardzoFajnyZespol

import cv2
import numpy as np
from functions.img_resize import image_resize

def add_padding(image, ww, hh):

    print("Image_Shape_start: " + str(image.shape))
    # ht, wd, cc = image.shape
    ht, wd = image.shape

    if ht > hh or wd > ww:
        if ht > wd:  # image is scaled maintaining its aspect ratio
            image = image_resize(image, height=hh)
        else:
            image = image_resize(image, width=ww)
    else:  # if image is smaller than img_dims then it will be stretched to match, temporary solution
        print("WARNING: This image is smaller than img_dims!\n img_dims: " + str(ww) + ", " + str(hh) + "\n This image: " + str(wd) + ", " + str(ht))
        image = cv2.resize(image, (ww, hh))

    ht, wd = image.shape
    print("Image_Shape_scaled: " + str(image.shape))
    # creating background

    color = (0)  # padding background color
    result = np.full((hh, ww), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy image into center of result image
    result[yy:yy + ht, xx:xx + wd] = image
    return result
    # view result
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
