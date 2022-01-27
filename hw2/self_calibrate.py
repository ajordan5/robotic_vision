import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import time
from calibrate import Camera_calib


camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
winSize = (height, width)

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
CORNER = 3
LINE = 4
ABSDIFF = 5
ABSDIFF_ONCE = 6

def cvMat2tkImg(arr):           # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

class App(Frame):
    def __init__(self, winname='OpenCV'):       # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        self.calibrator = Camera_calib(filepath="./hw2/my_photos/", fileout="./hw2/webcam_params.txt", boardsize=(9,7))

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady = 2)
        # Picture counter
        self.display = Label(self.root, text="", )
        self.display.pack(side="right")

        # Timer
        self.start = time.time()
        self.wait_time = 1
        self.counter = 0
        self.num_images = 40
        
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()
        # Store a previous frame for differencing
        _, self.previous_frame = camera.read()
        self.current_frame = self.previous_frame.copy()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                if self.timer() > self.wait_time:
                    cv.imwrite("./hw2/my_photos/AR{}.jpg".format(self.counter), frame)
                    self.counter+=1
                    self.reset_time()
                    self.display.configure(text="{} pictures taken".format(self.counter))

                image = cvMat2tkImg(cv.flip(frame, 1))
                self.panel.configure(image=image)
                self.panel.image = image

                if self.counter>self.num_images:
                    self.display.configure(text="Calibrating")
                    mtx, dist = self.calibrator.calibrate()
                    self.undistort(mtx, dist)
                    self.startstop()

    def undistort(self, K, D):
        image = cv.imread("./hw2/my_photos/AR1.jpg")
        h,w = image.shape[:2]
        K = np.array(K)
        D = np.array(D)
        new_K, roi = cv.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
        frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        frame_undistort = cv.undistort(frame, K, D, None, new_K)
        diff = cv.absdiff(frame, frame_undistort)
        combined = np.concatenate((frame, frame_undistort, diff), axis=1)
        cv.imshow("Original, Undistorted, Difference", combined)
        name = "./hw2/webcam_diff.jpg"
        cv.imwrite(name, combined)
        cv.waitKey()


    def timer(self):
        return time.time() - self.start

    def reset_time(self):
        self.start = time.time()

    def startstop(self):        #toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):              #run main loop
        self.root.mainloop()

    def exitApp(self):          #exit loop
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
#release the camera
camera.release()
cv.destroyAllWindows()