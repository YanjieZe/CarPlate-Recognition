import cv2
import numpy as np


def show_img(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
    cv2.imshow("img", img)
    cv2.waitKey(0)

def save_img(name, img):
    cv2.imwrite(name, img)