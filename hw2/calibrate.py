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

class Camera_calib:
    """Calibrate a camera given a set of pictures of a chessboard.
    
    Note: All images must start with "AR*" (i.e. "AR1.jpg") 
    
    Params:
        filepath (string): filepath to source images
        fileout (string): path and name for output txt with camera parameters
        boardsize (tuple(int)): dimensions of inner chessboard, do not include the outer squares of the chessboard
        pixels_per_mm (int): Numper of pixels per mm. This can be calculated by taking the number of pixels in the x or y direction
                                divided by the actual size in that direction. Example a 648x488 pixel camera with a sensor size of 
                                4.8mmx3.6mm -> pixels_per_mm = 648/4.8 = 135"""
                                
    def __init__(self, filepath="./hw2/images/", fileout="./hw2/camera_params.txt", boardsize = (10,7), pixels_per_mm=135) -> None:
        # Lists for image and object points on chessboard 
        self.image_pts = []
        self.object_pts = []

        self.filepath = filepath
        # criteria for corner detection
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)

        # chessboard
        self.width = boardsize[0]
        self.height = boardsize[1]

        # Template for object points
        temp = self.width*self.height
        self.pts = np.zeros((temp,3), np.float32)
        self.pts[:,:2] = np.mgrid[0:self.width,0:self.height].T.reshape(-1,2)
        # file to write params
        self.fileout = fileout

    def calibrate(self):
        self.load_images()
        # Find corners in each image
        for image, name in zip(self.images, self.filenames):
            frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(frame, (self.width, self.height))
            
            # If found pattern, add to object points and image points
            if ret:
                self.object_pts.append(self.pts)
                corners = cv.cornerSubPix(frame, corners, (5,5), (-1,-1), self.criteria)
                self.image_pts.append(corners)
            else:
                print("corners not found for {}".format(name))

        # Calibrate based on corners found
        image_pts = np.array(self.image_pts)
        object_pts = np.array(self.object_pts)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_pts, image_pts, frame.shape[::-1], None, None)
        fx = mtx[0,0]/135
        fy = mtx[1,1]/135

        print("K matrix:", mtx, "\n\n")
        print("Distortion params:", dist, "\n\n")
        print("Focal point in mm (x,y)", fx, fy)

        file = open(self.fileout, "w")
        file.write("K = {}\n".format(mtx))
        file.write("D = {}\n".format(dist))
        file.write("fx = {}\n".format(fx))
        file.write("fy = {}\n".format(fy))
        return mtx, dist

    def load_images(self):
        # Extract images
        print(os.getcwd())
        self.filenames = [file for file in os.listdir(self.filepath) if file.startswith("AR")]
        self.images = [cv.imread(self.filepath + imagename) for imagename in self.filenames]

        


