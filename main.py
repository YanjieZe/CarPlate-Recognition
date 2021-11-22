from operator import pos
import numpy as np
import cv2
import utils
from localization import PlateLocation, draw_box
from separation import get_area_img, postprocess
from template_match import TemplateMatcher
import os

if __name__=='__main__':
    level = 'difficult'
    root_path = 'requirement_images'

    if level=='difficult':
    
        params =  [( os.path.join(root_path, level ,"3-1.jpg"), '沪EWM957', "blue", 0.1, 0.1, 0.1),
                 ( os.path.join(root_path, level, "3-2.jpg"), '沪ADE6598', "green", 0.1, 0.1, 0.1), 
                 ( os.path.join(root_path, level, "3-3.jpg"), '皖SJ6M07', "blue", 0.1, 0.1, 0.1)]
        for img_path, gt, color,match_threshold,scale_speed, iou_thres in params:
            
            print('--------- test %s -------'%img_path)
            img = cv2.imread(img_path)

            location  = PlateLocation(img, plate_color=color)
            sheared_img = get_area_img(origin_img=img, location=location)
            utils.show_img(sheared_img)
            post_img = postprocess(sheared_img, color)
            

            matcher = TemplateMatcher(threshold=match_threshold, template_path=os.path.join('templates', level))
            # result = matcher.match(test_img, test_template)
            plate_number = matcher.loop_match(post_img, scale_speed, iou_thres)
            print("pred:", plate_number)
            print("ground truth:", gt)

    elif level=='medium':
        params =  [( os.path.join(root_path, level ,"2-1.jpg") ,'沪EWM957',  "blue", 0.1, 0.1, 0.1),
                 ( os.path.join(root_path, level, "2-2.jpg"), '豫B20E68',"blue", 0.1, 0.1, 0.1), 
                 ( os.path.join(root_path, level, "2-3.jpg"), '沪A93S20', "deep blue", 0.1, 0.1, 0.1)]

        for img_path,gt, color,match_threshold,scale_speed, iou_thres,fake in params:
            
            print('--------- test %s -------'%img_path)
            img = cv2.imread(img_path)

            location  = PlateLocation(img, plate_color=color)
            sheared_img = get_area_img(origin_img=img, location=location, use_shear=False)

            # utils.show_img(sheared_img)
            post_img = postprocess(sheared_img, color)

            matcher = TemplateMatcher(threshold=match_threshold, template_path=os.path.join('templates', level))
            # result = matcher.match(test_img, test_template)
            plate_number = matcher.loop_match(post_img, scale_speed, iou_thres)
            print("pred:", plate_number)
            print("ground truth:", gt)

    elif level=='easy':
        params =  [( os.path.join(root_path, level ,"1-1.jpg") ,'沪EWM957', "blue", 0.2, 0.1, 0.2),
                 ( os.path.join(root_path, level, "1-2.jpg"), '沪AF02976', "green", 0.2, 0.1, 0.2), 
                 ( os.path.join(root_path, level, "1-3.jpg"), '鲁NBK268',"blue", 0.2, 0.1, 0.2)]
        for img_path, gt, color,match_threshold,scale_speed, iou_thres in params:
            
            print('--------- test %s -------'%img_path)
            img = cv2.imread(img_path)

            

            matcher = TemplateMatcher(threshold=match_threshold, template_path=os.path.join('templates', level))
            # result = matcher.match(test_img, test_template)
            plate_number = matcher.loop_match(img, scale_speed, iou_thres)
            print("pred:", plate_number)
            print("ground truth:", gt)



