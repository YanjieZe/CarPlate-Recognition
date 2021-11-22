from operator import pos
import numpy as np
import cv2
import utils
from localization import PlateLocation, draw_box
from separation import get_area_img, postprocess
from template_match import TemplateMatcher

if __name__=='__main__':
    # for img_path,color in [("test2.jpg","green")]:
    params =  [("test1.jpg", "blue", 0.1, 0.1, 0.1), ("test2.jpg","green", 0.1, 0.1, 0.1), ("test3.jpg","blue", 0.1, 0.1, 0.1)]
    for img_path,color,match_threshold,scale_speed, iou_thres in params:
        
        print('--------- test %s -------'%img_path)
        img = cv2.imread(img_path)

        location  = PlateLocation(img, plate_color=color)
        sheared_img = get_area_img(origin_img=img, location=location)
        post_img = postprocess(sheared_img, color)
        

        matcher = TemplateMatcher(threshold=match_threshold)
        # result = matcher.match(test_img, test_template)
        plate_number = matcher.loop_match(post_img, scale_speed, iou_thres)
        print(plate_number)


