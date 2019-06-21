"""
    Testing bed for camera functions.
"""

import camera

myCamera = camera.Camera(2, 600, 800, 60)

myCamera.record("example.avi")

myCamera.release_cam()