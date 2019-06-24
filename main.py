"""
    Testing bed for camera functions.
"""

import camera

myCamera = camera.Camera(2, 600, 800)

myCamera.classify_live()

myCamera.release_cam()