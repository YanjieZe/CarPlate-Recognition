from typing import IO
import numpy as np
import cv2
from numpy.lib.shape_base import tile
import utils
import matplotlib.pyplot as plt
import os
import copy


class TemplateMatcher:
    def __init__(self, threshold=0.1, template_path='templates', visualize=False):
        self.template_path = template_path
        self.methods = ['cv2.TM_CCOEFF',  \
         'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        self.threshold = threshold
        self.visualize = visualize

    def match(self, img, template, scale_speed):
        """
        Multi scale Template match
        """


        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        template = cv2.Canny(template, 50, 200)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # utils.show_img(gray)
        found = None
        visualize = self.visualize

        (tH, tW) = template.shape[:2]

        # find scale
        scale_low = 0.1
        scale_high = 1
        scale_num = 20
        # import pdb; pdb.set_trace()
        while(gray.shape[0]*scale_low < tH):
            scale_low += scale_speed
        scale_high = scale_low + 1.0

        # loop over the scales of the image
        for scale in np.linspace(scale_low, scale_high, scale_num)[::-1]:

            resized = cv2.resize(gray, ( int(gray.shape[1]*scale),int(gray.shape[0] * scale) ) )
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                # print('resize shape too small, break')
                break
            
            edged = cv2.Canny(resized, 50, 200)
            
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if visualize:

                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        if found is None:
            if self.visualize:
                print('Not found any. Match failure.')
            return None
            
        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        if self.visualize:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

        if self.visualize:
            print('score is:', maxVal)
            if maxVal > self.threshold:
                print('match success.')
            else:
                print('match failure.')
        
        return found


    def template_dataset(self):
        """
        Return the templates dataset
        """
        self.template_file_list = os.listdir(self.template_path)
        template_dataset = []
        

        for template_file in self.template_file_list:
            if template_file=='.DS_Store':
                continue
            img = cv2.imread(os.path.join(self.template_path, template_file))
            # parse file name
            label = template_file.split('.')[0]
            label = label.split('_')[0]
            template_dataset.append( (label, img) )

        
        return template_dataset


    def loop_match(self, plate_img, scale_speed, iou_threshold):

        debug = False

        template_dataset = self.template_dataset()
        result_template_list = []
        img = copy.deepcopy(plate_img)
        img_withnms= copy.deepcopy(plate_img)
        # loop over
        for data in template_dataset:
            label = data[0]
            template = data[1]
            try:
                result = self.match(plate_img, template, scale_speed)
            except:
                import pdb; pdb.set_trace()
                print('Error here.')

            if result is None:
                continue
            else:
                (tH, tW) = template.shape[:2]
                (maxVal, maxLoc, r) = result
                   
                if maxVal > self.threshold:
                    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
                    (midX, midY) = (int(startX/2+endX/2), int(startY/2+endY/2))
                    if debug:
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

                    result_template_list.append((label, (startX, startY, endX, endY), maxVal))
                    # result_template_list.append((label, (midX,midY)))
                    # result_template_list.append((label, (endX,endY)))
        
        # non-maximum suppression
        result_template_list = self.NMS(result_template_list, iou_threshold=iou_threshold)

        # filter
        def compare(a):
            return a[1][0]

        result_template_list.sort(key=compare, reverse=False)

        if debug:
            utils.show_img(img)
            # utils.save_img('report_img/without_nms.png', img)

            for box in result_template_list:
                label, (startX, startY, endX, endY), maxVal = box
                cv2.rectangle(img_withnms, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
            utils.show_img(img_withnms)
            # utils.save_img('report_img/with_nms.png', img_withnms)
            # print(result_template_list)
        
        output = []
        for i in range(len(result_template_list)):
            output.append(result_template_list[i][0])

        output = "".join(output)

        return output

    @staticmethod
    def compute_iou(gt_box,b_box):
        '''
        计算iou
        :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
        :param b_box: bounding box b_box 表示形式同上
        :return: 
        '''
        width0=gt_box[2]-gt_box[0]
        height0 = gt_box[3] - gt_box[1]
        width1 = b_box[2] - b_box[0]
        height1 = b_box[3] - b_box[1]
        max_x =max(gt_box[2],b_box[2])
        min_x = min(gt_box[0],b_box[0])
        width = width0 + width1 -(max_x-min_x)
        max_y = max(gt_box[3],b_box[3])
        min_y = min(gt_box[1],b_box[1])
        height = height0 + height1 - (max_y - min_y)
    
        interArea = width * height
        boxAArea = width0 * height0
        boxBArea = width1 * height1
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou
        



    @staticmethod
    def NMS(box_list, iou_threshold=0.4):
        """
        Non-maximum suppression
        """

        result_list = []

        def val_compare(a):
            return a[2]

        box_list.sort(key=val_compare, reverse=True)
        while(box_list!=[]):
            tmp_box = box_list.pop(0)
            result_list.append(tmp_box)
            for tmp_box2 in box_list:
                IOU = TemplateMatcher.compute_iou(tmp_box[1], tmp_box2[1])
                if IOU > iou_threshold:
                    box_list.remove(tmp_box2)

        
        return result_list



        
    
if __name__=='__main__':

    threshold = 0.01

    test_img_path = 'test1.jpg_plate.jpg' # 沪EWM957
    # test_img_path = 'test2.jpg_plate.jpg' # 沪ADE6598
    # test_img_path = 'test3.jpg_plate.jpg' # 皖SJ6M07

    # test_template_path = 'templates/D.png'
    
    test_img = cv2.imread(test_img_path)

    # test_template = cv2.imread(test_template_path)

    matcher = TemplateMatcher(threshold=threshold, visualize=False)
    # result = matcher.match(test_img, test_template, 0.1)

    plate_number = matcher.loop_match(test_img, scale_speed=0.3, iou_threshold=0.05)
    # print(plate_number)

