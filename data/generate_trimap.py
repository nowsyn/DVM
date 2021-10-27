import cv2
import numpy as np


def gen_trimap(alpha, ksize=3, iteration=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iteration)
    eroded = cv2.erode(alpha, kernel, iterations=iteration)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255                                                                                                                            
    trimap[dilated <= 0] = 0
    return trimap
