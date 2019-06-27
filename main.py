"""
    Testing bed for camera / network functionality.
    Currently loads network weights from a saved file(nn_weights.h5).
"""

import camera

# Instantiate a Camera object: (device number, height, width, window-of-interest size).
cam = camera.Camera(2, 600, 800, 400)

# Camera object loads provided weight file(.h5).
cam.load_network()

# Starts a live feed for digit classification.
cam.classify_live()

# Releases Camera object when finished.
cam.release_cam()