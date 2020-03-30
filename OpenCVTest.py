import cv2
import numpy as np
from display import Display

W = 1920//2
H = 1080//2

disp = Display(W, H)

class FeatureExtractor(object):
    def __init__(self):
        orb = cv2.ORB_create(100) # pylint: disable=maybe-no-member

    def extract(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3) # pylint: disable=maybe-no-member 
        return feats

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (W,H)) # pylint: disable=maybe-no-member
    keypoints = fe.extract(img) # collect array of keypoints

    for point in keypoints: # iterate keypoints array
        u,v = map(lambda n: int(round(n)), point[0]) # extract keypoints coords
        cv2.circle(img, (u,v), color=(0,255,0), radius=3) # pylint: disable=maybe-no-member

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