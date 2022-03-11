import cv2
import numpy as np


class BackgroundSubtractor:

    def __init__(self, min_accuracy=0.1, min_blend_area=3, kernel_fill=20, dist_threshold=15000, history=60, bs_type="MOG2"):
        self.min_accuracy = max(min_accuracy, 0.7)
        self.min_blend_area = min_blend_area
        self.kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.kernel_fill = np.ones((kernel_fill,kernel_fill), np.uint8)
        self.dist_threshold = dist_threshold
        self.history = history
        self.bs_type = bs_type

    def create_background(self):
        if self.bs_type == "MOG2":
            self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=self.history)
        else:
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=self.history, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
        return self.fgbg

