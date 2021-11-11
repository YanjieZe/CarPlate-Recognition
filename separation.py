import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils 
from localization import PlateLocation

def get_area_img(origin_img, location):
    """
    Clip the origin img and get the plate
    """
    pos_x,pos_y = location[0]
    angle = location[2]
    reverse_shear = False
    if angle<-45:
        reverse_shear = True
        height, width = location[1]
    else:
        width,height = location[1]

    if angle < -45.0:
        angle += 90
    # rotation
    RotationMatrix = cv2.getRotationMatrix2D((pos_x,pos_y),angle,1)
    img_rotated = cv2.warpAffine(origin_img, RotationMatrix, (origin_img.shape[1], origin_img.shape[0]))
    # utils.show_img(img_rotated)

    # clip
    img_cliped = img_rotated[int(pos_y-height/2):int(pos_y+height/2), int(pos_x-width/2):int(pos_x+width/2)]

    
    
    # shear
    if reverse_shear:
        ShearMatrix = np.array([[1, -0.3,0],
                            [0, 1,0]])
    else:
        ShearMatrix = np.array([[1, 0.3,0],
                                [0, 1,0]])
    img_sheared = cv2.warpAffine(img_cliped, ShearMatrix, (img_cliped.shape[1]+10, img_cliped.shape[0]+10))

    # utils.show_img(img_sheared)
    return img_sheared

def get_plate_character(plate_img):
    # 将图像灰度化
    img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.dilate(img, kernel)
    
    # 轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    img_draw = cv2.drawContours(plate_img.copy(), contours, -1, (255,255,0))
    utils.show_img(img_draw)
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    words = sorted(words,key=lambda s:s[0],reverse=False)

    i = 0
    for word in words:
        # 根据轮廓的外接矩形筛选轮廓
        if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
            i = i+1
            splite_image = plate_img[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            splite_image = cv2.resize(splite_image,(25,40))
            word_images.append(splite_image)

    for i,j in enumerate(word_images):
        plt.subplot(1,7,i+1)
        plt.imshow(word_images[i],cmap='gray')

    # utils.show_img(img)

if __name__=='__main__':
    img_path = "test1.jpg"
    img = cv2.imread(img_path)
    location = PlateLocation(img, 'blue')
    img_sheared = get_area_img(origin_img=img, location=location)

    get_plate_character(img_sheared)
    