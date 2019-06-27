'''
    This file defines the class object Camera.

    A Camera object instantiates the following:
        -A video feed of specified height, width, and device number. Also includes a variable for "window of interest".
        -A neural network trained on the MNIST dataset(can either train one or load saved weights).

    Input from the camera is transformed from a "window of interest" into "MNSIT-style" data before being classified by 
    the network. 
'''

import cv2 as cv
import network as nn

class Camera:
    def __init__(self, device_no, height, width, capt_size):
        # Camera Properties
        self.camera = cv.VideoCapture(device_no)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)

        # Image capturing properties. Values are used to calculate window of interest and write text onto image
        self.height = height
        self.width = width
        self.capt_size = capt_size
        self.text_pos = (10, height - 10)

        # Network properties. Network is None until instantiated. Network is trained or weights are loaded in from a file.
        self.network = None

    # Create and train network using MNIST.
    def nn_train(self, epochs):
        (td, tl), (vd, vl) = nn.MNIST_normalized()
        self.network = nn.get_trained_network(td, tl, vd, vl, epochs)

    # Create network and load existing weights from file(designated in network.py).
    def load_network(self):
        self.network = nn.create_network(784, 100, 10)
        nn.load_network_weights(self.network)

    def save_network(self):
        if self.network != None:
            nn.save_network_weights(self.network)

    '''
    # Starts a live camera feed featuring a designated capture window(capt_size).
    # Network will attempt to classify a frame every half-second.
    # Exit camera feed by pressing 'Q'.
    '''
    def classify_live(self):
        frame_count = 0
        outcome = None
        while(self.camera.isOpened()):
            ret, frame = self.camera.read()
            if ret == True:
                data_NN = self.convert_to_data(frame, self.capt_size)
                self.center_rect(frame, self.capt_size)
                cv.putText(frame, "Predicted Digit: " + str(outcome), self.text_pos, 2, 1, (0, 0, 100), 2, cv.LINE_AA)
                cv.imshow("Camera Feed. Press 'Q' to quit.", frame)
                cv.imshow("MNIST Form", data_NN)
                if frame_count % 15 == 0:
                    data_NN = data_NN.reshape(data_NN.shape[0] * data_NN.shape[1]) / 255.0
                    outcome = nn.single_prediction(data_NN, self.network)
                    cv.putText(frame, "Predicted Digit: " + str(outcome), self.text_pos, 2, 1, (0, 0, 100), 2, cv.LINE_AA)
                frame_count += 1
                if frame_count == 30:
                    frame_count = 0
                
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    
    # Releases camera feed.
    def release_cam(self):
        self.camera.release()

    # Wrapper function for cropping image. Uses tuples for TOP-left and BOTTOM-right corners.
    def crop_image(self, top, bottom, img):
        return img[top[1]:bottom[1], top[0]:bottom[0]]

    # Wrapper function for cropping square center of image. Calculates corners used in previous crop_image(). 
    def center_crop(self, img, square_side):
        half_side = int(square_side/2)
        top = (int(self.width/2)  - half_side, int(self.height/2) - half_side)
        bottom = (int(self.width/2) + half_side, int(self.height/2) + half_side)
        return self.crop_image(top, bottom, img)

    '''
        Funtion that crops center of captured image and transforms it into something that resembles an example from MNIST.
        An image made using this is then put into the neural network.
        Note: lighting has the greatest effect on the creation of data examples. Thresholding must be adjusted to compensate for it.    
    '''
    def convert_to_data(self, img, square_side):
        cropped = self.center_crop(img, square_side)
        cropped = cv.resize(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY), (28,28))
        cropped = 255 - nn.np.where(cropped > 200, 255, cropped)
        return nn.np.where(cropped < 150, 0, cropped)
    
    # Wrapper function for OpenCV's rectangle function that calculates square corners used to draw a window of interest
    # where network data is captured and transformed.
    def center_rect(self, img, square_side):
        half_side = int(square_side/2)
        top = (int(self.width/2)  - half_side, int(self.height/2) - half_side)
        bottom = (int(self.width/2) + half_side, int(self.height/2) + half_side)
        return cv.rectangle(img, top, bottom, (0, 0, 255), 3)

    