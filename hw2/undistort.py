import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import LabelFrame, Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import glob
from camera_params import K, D
import os
print(os.getcwd())

filepath = "./hw2/images/"
filenames = [file for file in os.listdir('./hw2/images') if not file.startswith("AR")]
images = [cv.imread(filepath + imagename) for imagename in filenames]
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
h,w = images[0].shape[:2]
K = np.array(K)
D = np.array(D)
new_K, roi = cv.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))

for frame, name in zip(images, filenames):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_undistort = cv.undistort(frame, K, D, None, new_K)
    diff = cv.absdiff(frame, frame_undistort)
    combined = np.concatenate((frame, frame_undistort, diff), axis=1)
    cv.imshow("Original, Undistorted, Difference", combined)
    name = "hw2/abs_diff_" + name
    cv.imwrite(name, combined)
    cv.waitKey()



