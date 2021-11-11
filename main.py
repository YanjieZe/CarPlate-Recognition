import numpy as np
import cv2
import utils
from localization import PlateLocation, draw_box
from separation import get_area_img, postprocess

if __name__=='__main__':
    # for img_path,color in [("test2.jpg","green")]:
    for img_path,color in [("test1.jpg", "blue"), ("test2.jpg","green"), ("test3.jpg","blue")]:
        
        print('--------- test %s -------'%img_path)
        img = cv2.imread(img_path)

        location  = PlateLocation(img, plate_color=color)
        sheared_img = get_area_img(origin_img=img, location=location)
        post_img = postprocess(sheared_img, color)
        

        utils.save_img("%s_plate.jpg"%img_path, post_img)

        # utils.show_img(post_img)


