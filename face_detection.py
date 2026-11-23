"""Viola-Jones face detection using OpenCV's pretrained Haar cascade.


"""

import sys
import cv2

NAME = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + NAME)

if cascade.empty():
    cascade = cv2.CascadeClassifier(NAME)
if cascade.empty():
    sys.exit(f"Could not load {NAME}. Try again")

