import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt

class TemplateMatcher:
    def __init__(self, template_path='templates'):
        self.template_path = template_path
        self.methods = ['cv2.TM_CCOEFF',  \
         'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        self.threshold = 0.90
        self.visualize = True

    def match(self, img, template, method_idx=0):
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

        # loop over the scales of the image
        for scale in np.linspace(0.1, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = cv2.resize(gray, ( int(gray.shape[1]*scale),int(gray.shape[0] * scale) ) )
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
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
        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # draw a bounding box around the detected result and display the image
        if visualize:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imshow("Image", img)
            cv2.waitKey(0)


        print('score is:', maxVal)
        if maxVal > self.threshold:
            print('match success.')
        else:
            print('match failure.')

        
        return found

    
if __name__=='__main__':
    test_img_path = 'test1.jpg_plate.jpg' # 沪EWM957
    test_template_path = 'templates/E.png'
    
    test_img = cv2.imread(test_img_path)
    test_template = cv2.imread(test_template_path)

    matcher = TemplateMatcher()
    result = matcher.match(test_img, test_template)


