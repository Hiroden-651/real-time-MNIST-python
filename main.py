"""
    Testing bed for camera / network functions.
    Currently loads network weights from a saved file(nn_weights.h5).
"""

import camera

myCamera = camera.Camera(2, 600, 800, 400)

myCamera.load_network()
myCamera.classify_live()
myCamera.release_cam()