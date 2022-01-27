import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
winSize = (height, width)
videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format

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
        global btnCapture
        btnCapture = Button(text="Capture", command=self.diff_capture)
        btnCapture['font'] = helv18
        btnCapture.pack(side='right', pady = 2)
        self.capture1 = False
        self.capture2 = False
        self.absdiff = False
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(255)
        Slider1 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(0)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Edge", variable=mode, value=EDGE).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Corner", variable=mode, value=CORNER).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
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
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if mode.get() == BINARY:
                    # Binarize
                    frame = cv.inRange(frame_gray, Slider1.get(), Slider2.get())

                elif mode.get() == EDGE:
                    # Convert to greyscale and blur for edge detection
                    frame_blur = cv.blur(frame_gray, (3,3))
                    edges = cv.Canny(frame_blur, Slider1.get(), Slider2.get())
                    mask = edges != 0
                    frame = frame * (mask[:,:,None].astype(frame.dtype))

                elif mode.get() == CORNER:
                    # Convert to greyscale and find corners
                    corners = cv.goodFeaturesToTrack(frame_gray, Slider1.get(), .01, 10, None, \
                                blockSize=3, gradientSize=3, useHarrisDetector=False, k=.04)

                    # Refined corner locations
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
                    corners = cv.cornerSubPix(frame_gray, corners, (5,5), (-1, -1), criteria)
                    for i in range(corners.shape[0]):
                        cv.circle(frame, (int(corners[i,0,0]), int(corners[i,0,1])), 4, (124,252,0), cv.FILLED)


                elif mode.get() == LINE:
                    #detect edges
                    edges = cv.Canny(frame_gray, 70, 255, None, 3)
                    cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 5+Slider1.get(), None, 50, 10+Slider2.get())

                    if lines is not None:
                        for i in range(0, len(lines)):
                            l = lines[i][0]
                            cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                    frame = cdst
                    
                elif mode.get() == ABSDIFF:
                    self.current_frame = frame
                    frame = cv.absdiff(self.current_frame, self.previous_frame)
                    self.previous_frame = self.current_frame
                    
                elif mode.get() == ABSDIFF_ONCE:
                    if not self.absdiff:
                        text = Label(self, text="Press capture twice to get two images and the difference")
                        text.pack(side='top')
                        self.absdiff = True
                    frame = self.diff(frame)
                    

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(frame)

    def diff(self, frame):
        if self.capture2:
            frame = self.difference

        return frame

    
    def diff_capture(self):
        if mode.get() != ABSDIFF:
            text = Label(self, text="Capture only used for absdiff")
            text.pack(side='top')
        else:
            if not self.capture1:
                _, self.image1 = camera.read()
                text = Label(self, text="First image captured, press again to get the next image")
                text.pack(side='top')
                self.capture1 = True
            else:
                _, self.image2 = camera.read()
                text = Label(self, text="Second image captured, difference displayed")
                text.pack(side='top')
                self.difference = cv.absdiff(self.image1, self.image2)
                self.capture2 = True

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