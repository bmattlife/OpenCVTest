import cv2
import numpy as np
from display import Display
from extractor import Extractor

W = 1920//3
H = 1080//3

disp = Display(W, H)
fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W,H)) # pylint: disable=maybe-no-member
    matches = fe.extract(img) # collect array of features
    if matches is None:
        return

    for pt1, pt2 in matches: # iterate features array
        u1,v1 = map(lambda n: int(round(n)), pt1.pt) # extract features coords
        u2,v2 = map(lambda n: int(round(n)), pt2.pt)
        #cv2.circle(img, (u1,v1), color=(0,255,0), radius=1) # pylint: disable=maybe-no-member
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    for m in matches:
        print(m)

    disp.draw(img)

def brightness(pixel):
    B = pixel[0]
    G = pixel[1]
    R = pixel[2]
    return 0.2126*R + 0.7152*G + 0.0722*B

if __name__ == "__main__":
    cap = cv2.VideoCapture("test2.mp4") # pylint: disable=maybe-no-member

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break