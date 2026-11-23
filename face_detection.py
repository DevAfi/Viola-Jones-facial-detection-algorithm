"""Viola-Jones face detection using OpenCV's pretrained Haar cascade."""

import sys
import cv2

NAME = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + NAME)

if cascade.empty():
    cascade = cv2.CascadeClassifier(NAME)
if cascade.empty():
    sys.exit(f"Could not load {NAME}. Try again")


def detect(frame):
    """Return the frame with a box drawn around the face"""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Greyscale
    faces = cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,  # window is scaled up by 10% each pass
        minNeighbors=5,  # overlapping detections needed to keep a face (merge step)
        minSize=(30, 30),  # ignore windows smaller than this
    )
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame

def run_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not find {path}")
    out = detect(img)
    out_path = path.rsplit(".", 1)[0] + "_out.jpg"
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path}")
    cv2.imshow("Face detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()