import numpy as np
import cv2
import utils
from localization import PlateLocation, draw_box
from separation import get_area_img

if __name__=='__main__':
    # for img_path,color in [("test2.jpg","green")]:
    for img_path,color in [("test1.jpg", "blue"), ("test2.jpg","green"), ("test3.jpg","blue")]:
        
        print('--------- test %s -------'%img_path)
        img = cv2.imread(img_path)

        location  = PlateLocation(img, plate_color=color)

        # #for localization
        # draw_box(img, location, "%s_result"%img_path)

        # print('The location of the plate:')
        # print(' position x and y:', location[0][0],location[0][1])
        # print(' width and height:', location[1][0], location[1][1])
        # print(' rotation angle:', location[2])

        sheared_img = get_area_img(origin_img=img, location=location)
        utils.save_img("%s_plate.jpg"%img_path, sheared_img)
        # utils.show_img(sheared_img)


