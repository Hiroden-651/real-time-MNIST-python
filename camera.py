'''
    Bradley Maness
    Personal Project
    June 20th, 2019
    Personal Deadline: June 27th, 2019

    This file will define a class object for manipulating the camera.
'''

import sys
import cv2 as cv

class Camera:
    def __init__(self, device_no, height, width, framerate):
        self.camera = cv.VideoCapture(device_no)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv.CAP_PROP_FPS, framerate)

        self.height = height
        self.width = width
        self.fps = framerate

    def record(self, filename):

        outfile = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))

        while(self.camera.isOpened()):
            ret, frame = self.camera.read()
            if ret == True:
                cv.imshow("frame", frame)
                outfile.write(frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
        outfile.release()
    
    def release_cam(self):
        self.camera.release()