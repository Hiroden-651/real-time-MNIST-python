'''
    This file will define a class object for manipulating the camera.
'''

import sys
import cv2 as cv
# import a neural network trained on MNIST

class Camera:
    def __init__(self, device_no, height, width):
        self.camera = cv.VideoCapture(device_no)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)

        self.height = height
        self.width = width

    def record(self, filename):

#        outfile = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))

        while(self.camera.isOpened()):
            ret, frame = self.camera.read()
            if ret == True:

                # DO STUFF

                cv.imshow("RAW INPUT", frame)
                cv.imshow("MNIST FORM", self.convert_to_data(frame, 200))
#                outfile.write(frame)



                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
 #       outfile.release()

    def classify_live(self):
        frame_count = 0
        while(self.camera.isOpened()):
            ret, frame = self.camera.read()
            if ret == True:
                # DO STUFF

                data_NN = self.convert_to_data(frame, 200)
                self.center_rect(frame, 200)
                cv.imshow("Input", frame)
                cv.imshow("MNIST Form", data_NN

                if frame_count % 30 == 0:
                    print("Classify frame!")

                    # frame classification happens in here.

                frame_count += 1
                if frame_count == 30:
                    frame_count = 0

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    
    def release_cam(self):
        self.camera.release()

    def crop_image(self, top, bottom, img):
        return img[top[1]:bottom[1], top[0]:bottom[0]]

    def center_crop(self, img, square_side):
        half_side = int(square_side/2)
        top = (int(self.width/2)  - half_side, int(self.height/2) - half_side)
        bottom = (int(self.width/2) + half_side, int(self.height/2) + half_side)
        return self.crop_image(top, bottom, img)
        
    def convert_to_data(self, img, square_side):
        cropped = self.center_crop(img, square_side)
        return cv.resize(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY), (28,28))
    
    def center_rect(self, img, square_side):
        half_side = int(square_side/2)
        top = (int(self.width/2)  - half_side, int(self.height/2) - half_side)
        bottom = (int(self.width/2) + half_side, int(self.height/2) + half_side)
        return cv.rectangle(img, top, bottom, (0, 255, 0), 3)