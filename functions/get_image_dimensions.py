#Copyright (C) 2020 BardzoFajnyZespol

# import the necessary packages
import argparse
import cv2


refPt = []
cropping = False
sel_rect_endpoint = [(0, 0)]
ix, iy, = 0, 0

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, sel_rect_endpoint, ix, iy
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDBLCLK and not cropping:
        refPt = [(x, y)]
        cropping = True
        sel_rect_endpoint = [(x, y)]
        ix = x
        iy = y
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONDBLCLK and cropping:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        print("before:")
        print (refPt)
        refPt = [(min(ix, x), min(iy, y)), (max(ix, x), max(iy, y))]
        print("after:")
        print(refPt)
        cropping = False
        # draw a rectangle around the region of interest
        print(refPt[0])
        cv2.rectangle(get_crop_dims.image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", get_crop_dims.image)
        sel_rect_endpoint = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        sel_rect_endpoint = [(x, y)]

def get_crop_dims(im_address):
    get_crop_dims.image = cv2.imread(im_address)
    image = get_crop_dims.image
    clone = image.copy()
    height, width = image.shape[:2]
    win_height, win_width = 1080, 1920
    scale = 8
    if width > win_width:
        while width > win_width:
            width /= scale
            height /= scale
    elif height > win_height:
        while height > win_height:
            width /= scale
            height /= scale
    height, width = int(height), int(width)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)
    cv2.moveWindow('image', int(win_width/2 - width/2), int(win_height/2 - height/2))
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        if not cropping:
            cv2.imshow('image', image)
        elif cropping and sel_rect_endpoint:
            rect_cpy = image.copy()
            cv2.rectangle(rect_cpy, refPt[0], sel_rect_endpoint[0], (0, 255, 0), 1)
            cv2.imshow('image', rect_cpy)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        elif cv2.getWindowProperty('image', 0) == -1:
            break
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    x, y, w, h = 0, 0, 0, 0
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        x = refPt[0][0]
        y = refPt[0][1]
        if refPt[0][0] > refPt[1][0]:
            w = refPt[0][0] - refPt[1][0]
        else:
            w = refPt[1][0] - refPt[0][0]
        if refPt[1][1] > refPt[1][1]:
            h = refPt[0][1] - refPt[1][1]
        else:

            h = refPt[1][1] - refPt[0][1]

        print([refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]])
        print("Crop results: \n x: " + str(x) + "\n y: " + str(y) + "\n w: " + str(w) + "\n h: " + str(h))
        cv2.imshow("Cropped area", roi)
        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()
    return x,y,w,h
# address = "/Volumes/Karcioszka/Bosch/Work_copy/OK_copy/1.png"
# get_crop_dims(address)
