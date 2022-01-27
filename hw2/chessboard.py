import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import LabelFrame, Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import glob
import os
print(os.getcwd())

size = (9,7)
#videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format
filepath = "./hw2/my_photos/"
filenames = [file for file in os.listdir(filepath) if file.startswith("AR")]
images = [cv.imread(filepath + imagename) for imagename in filenames]

# Locate chessboard corners
chessboard = images[2]

frame = cv.cvtColor(chessboard, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(frame, size)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
cv.imshow("chessboard", frame)
cv.waitKey()
corners = cv.cornerSubPix(frame, corners, (5,5), (-1,-1), criteria)
cv.drawChessboardCorners(chessboard, size, corners, ret)
cv.imshow("chessboard", chessboard)
cv.imwrite("hw2/task1.jpg", chessboard)
print("Press a key to continue")
cv.waitKey()


