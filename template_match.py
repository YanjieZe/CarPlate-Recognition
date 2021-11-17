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
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = cv2.resize(gray, ( int(gray.shape[1]*scale),int(gray.shape[0] * scale) ) )
            # utils.show_img(resized)
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                print('resize shape too small, break')
                break
            
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # check to see if the iteration should be visualized
            if visualize:
                # draw a bounding box around the detected region
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


    def loop_match(self, plate_img, scale_speed):

        debug = True

        template_dataset = self.template_dataset()
        result_template_list = []
        img = copy.deepcopy(plate_img)
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

                    result_template_list.append((label, (startX,startY, endX, endY)))
                    # result_template_list.append((label, (midX,midY)))
                    # result_template_list.append((label, (endX,endY)))
        
        # filter
        def compare(a):
            return a[1][0]

        result_template_list.sort(key=compare, reverse=False)

        if debug:
            utils.show_img(img)
            utils.save_img('report_img/without_nms.png', img)
            print(result_template_list)
        
        output = []
        for i in range(len(result_template_list)):
            output.append(result_template_list[i][0])

        output = "".join(output)

        return output


        
    
if __name__=='__main__':

    threshold = 0.01

    # test_img_path = 'test1.jpg_plate.jpg' # 沪EWM957
    test_img_path = 'test2.jpg_plate.jpg' # 沪ADE6598
    # test_img_path = 'test3.jpg_plate.jpg' # 皖SJ6M07

    test_template_path = 'templates/D.png'
    
    test_img = cv2.imread(test_img_path)
    test_template = cv2.imread(test_template_path)

    matcher = TemplateMatcher(threshold=threshold, visualize=False)
    # result = matcher.match(test_img, test_template, 0.1)

    plate_number = matcher.loop_match(test_img, 0.3)
    # print(plate_number)

