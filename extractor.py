
import cv2
import numpy as np

class Extractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(100) # pylint: disable=maybe-no-member
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last = None
    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3) # pylint: disable=maybe-no-member 

        # extraction
        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        keypoints, descriptors = self.orb.compute(img, keypoints)

        # matches
        matches = None
        if self.last is not None:
            matches = self.bf.match(descriptors, self.last['descriptors'])
            matches = zip([keypoints[m.queryIdx] for m in matches], [self.last['keypoints'][m.trainIdx] for m in matches])

        # return
        self.last = {'keypoints': keypoints, 'descriptors': descriptors}
        return matches