import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import LabelFrame, Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import glob
from camera_params import K, D
import pandas

K = np.array(K)
D = np.array(D)
data = pandas.read_csv('hw2/data.txt', header=None)
img_pts = []
obj_pts = []

# Load feature pixel locations and 3d object points
for line in open('hw2/data.txt'):
    line = line.split()
    if len(line) == 2:
        img_pts.append(line)
    else:
        obj_pts.append(line)
img_pts = np.array(img_pts, np.float32)
obj_pts = np.array(obj_pts, np.float32)

# Use PNP solver to get a relative rotation and translation
ret, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, D)
print(rvec, tvec)

# Convert rotation to a 3x3 rot matrix
R =cv.Rodrigues(rvec)

file = open("./hw2/obj_pose.txt", "w")
file.write("t = {}\n".format(tvec))
file.write("r = {}\n".format(rvec))
file.write("R = {}".format(R[0]))