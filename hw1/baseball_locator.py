import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import LabelFrame, Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import glob
import time
import imutils

# Image directory
import os
from os import listdir

#videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format
filepath = "C:/Users/ajord/Documents/Masters Work/Winter_22/robotic_vision/hw1/baseball/*.jpg"
images = [cv.imread(file) for file in glob.glob(filepath)]

size = images[0].shape
height = size[0]
width = size[0]
winSize = (height, width)
videoout = cv.VideoWriter('./Video4.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format



def cvMat2tkImg(arr):           # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

class App(Frame):
    def __init__(self, winname='Baseball Detector'):
        
        self.root = Tk()
        self.stopflag = True

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        
        # Positions the window in the center of the page.
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first original frame
        image = cvMat2tkImg(images[0])
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="left")
        # Display binarized
        image = images[0]
        cv.threshold(image, 70, 255, cv.THRESH_BINARY)
        image = cvMat2tkImg(image)
        self.panel2 = Label(image=image)
        self.panel2.image = image
        self.panel2.pack(side="right")
        # buttons
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='bottom', pady = 2)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.find_baseball, args=())
        self.thread.start()
        img_grey = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
        self.previous_frame = cv.adaptiveThreshold(img_grey,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        self.current_frame = np.copy(self.previous_frame)
        # slider
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(15)
        Slider1 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(46)

    def find_baseball(self):
        
        while not self.stopevent.is_set():
            for orig in images:
                if not self.stopflag:
                    
                    #binarize the current image
                    #_, frame = cv.threshold(frame, 127, Slider2.get(), cv.THRESH_BINARY)
                    frame = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
                    frame = cv.medianBlur(frame, 1)
                    #frame = cv.blur(frame, (3,3))
                    #frame = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
                    
                    # difference with previous image
                    self.current_frame = frame
                    frame = cv.absdiff(self.current_frame, self.previous_frame)
                    self.previous_frame = self.current_frame
                    _, frame = cv.threshold(frame, Slider1.get(), Slider2.get(), cv.THRESH_BINARY)
                    #frame = cv.inRange(frame, Slider1.get(), Slider2.get())
                    #frame = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
                    

                    # find contours in the mask and initialize the current
                    # (x, y) center of the ball
                    rows = frame.shape[0]
                    circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT, 1, rows / 8,
                                            param1=46, param2=15,
                                            minRadius=7, maxRadius=22)
                    # only proceed if at least one contour was found
                    drawing = np.copy(orig)
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        
                        for i in circles[0, :]:
                            center = (i[0], i[1])
                            # circle center
                            cv.circle(drawing, center, 1, (0, 100, 100), 3)
                            # circle outline
                            radius = i[2]
                            cv.circle(drawing, center, radius, (255, 0, 255), 3)
                    time.sleep(0.1)
                    image = cvMat2tkImg(frame)
                    self.panel2.configure(image=image)
                    self.panel2.image = image
                    videoout.write(frame)
                    #grab an image
                    image = cvMat2tkImg(drawing)
                    self.panel.configure(image=image)
                    self.panel.image = image
                    



    def run(self):              #run main loop
        self.root.mainloop()

    def exitApp(self):          #exit loop
        self.stopevent.set()
        self.root.quit()
        
    def startstop(self):        #toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

app = App()
app.run()
#release the camera
videoout.release()
print("done")
cv.destroyAllWindows()