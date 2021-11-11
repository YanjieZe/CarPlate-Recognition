import numpy as np
import cv2
import matplotlib.pyplot as plt
WINDOW_NAME = "ZYJ's test"

debug = False

def show_img(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
    cv2.imshow("img", img)
    cv2.waitKey(0)

def draw_box(origin_img, location, img_name):
    # 可视化矩形
    # draw box
    box = cv2.boxPoints(location)
    for k in range(4):
        n1,n2 = k%4,(k+1)%4
        cv2.line(origin_img,(box[n1][0],box[n1][1]),(box[n2][0],box[n2][1]),(255, 0, 255),5)
    if debug:
        show_img(origin_img)
    cv2.imwrite('%s.jpg'%img_name, origin_img)


def preprocess(orig_img, plate_color):
    if plate_color not in ['blue', 'green']:
        raise Exception('Car Plate Color Error.')

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.blur(gray_img, (3, 3))

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)


    # use HSV to divide color
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    
    # 为绿色车牌设计的mask
    hsv_mask_for_img_green = (h > 60) & (h<70) & (s<90) & (v>150)

    # 为蓝色车牌设计的mask
    hsv_mask_for_img_blue = (h>90) &  (h<120) & (s>200) & (s<270) & (v>120) & (v<180)

    if plate_color=='blue':
        hsv_mask = hsv_mask_for_img_blue
    elif plate_color=='green':
        hsv_mask = hsv_mask_for_img_green

    hsv_mask = hsv_mask.astype(np.float64)
    
    mix_img = np.multiply(sobel_img, hsv_mask)

    mix_img = mix_img.astype(np.uint8)
    # show_img(mix_img)
    _, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.dilate(close_img, kernel, 5)

    return close_img

def rectangle_is_plate(rectangle):
    """
    判断矩形框是否可能是车牌
    """

    # some hyper parameters
    threshold_theta = 70
    threshold_hwratio = 10
    threshold_area = 5000

    pos, hw, theta = rectangle
    pox_x, pos_y = pos
    width, height = hw
    area = width * height

    if(theta>threshold_theta): # 角度过于倾斜
        return False

    if( height/(width+0.001)>threshold_hwratio or width/(height+0.001)>threshold_hwratio ): 
        # 长宽比不对
        return False

    if (area<threshold_area):
        # 矩形面积太小
        return False
    return True


def common_area(rec1, rec2):
    """
    用来计算两个旋转矩形的重叠面积
    
    """   
    section = cv2.rotatedRectangleIntersection(rec1, rec2)
    
    num_section = section[0]
    if num_section<1:
        return None # no intersection
    area = section[1]
    area = cv2.contourArea(area)
    # print('Common area:', area)
    return area


def bbox_compression(rectangle_list):
    """
    算法思想：针对有多个框在车牌周围，将他们整合起来。
    """
    new_rectangle_list = []
    threshold_intersection = 1000
    

    for i in range(len(rectangle_list)):
        for j in range(i+1, len(rectangle_list)):
            # 如果重叠面积达到阈值，将他们整合起来
            ca = common_area(rec1=rectangle_list[i], rec2=rectangle_list[j])
            if ca is not None and ca > threshold_intersection:
                points1 = cv2.boxPoints(rectangle_list[i])
                points2 = cv2.boxPoints(rectangle_list[j])
               
                point_list = []
                point_list.extend(points1)
                point_list.extend(points2)
                new_rect = cv2.minAreaRect(np.array(point_list))
                new_rectangle_list.append(new_rect)

    # 如果new-rectangle-list为空，直接返回 rectangle list
    if new_rectangle_list == []:
       return rectangle_list

    
    return new_rectangle_list

    
def get_plate_location(orig_img, img_processed):
    """
    对于预处理后的图片，查找其中可能是车牌的位置。
    返回可能性最大的那个矩形方框。
    """
    # get contour of processed img
    contours, heriachy = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_for_draw = orig_img.copy()

    possible_rec_list = [] # 存储可能的矩形框

    for i, contour in enumerate(contours):
        
        # draw contour
        # cv2.drawContours(img_for_draw, contours, i, (255, 0, 255), 5)
        
        # get bounding rectangle
        # minAreaRect返回最小外接矩形的：中心坐标(x,y)，宽高(w,h)，旋转角度(theta)
        # 注意，theta是度数，不是弧度
        rotate_rect = cv2.minAreaRect(contour)
        if rectangle_is_plate(rotate_rect):
            # 把可能的矩形框存储起来
            possible_rec_list.append(rotate_rect)

        # 进行整合

    # compress
    possible_rec_list = bbox_compression(possible_rec_list)


    if possible_rec_list==[]:
        print('Fail to find the location box')
        return []

    # 选择最大的矩形来返回，因为我们假设一张图片中只有一个车牌
    max_area = 0
    max_rec = None
    for rec in possible_rec_list:
        pos, hw, theta = rec
        pox_x, pos_y = pos
        width, height = hw
        area = width * height
        if area > max_area:
            max_area = area
            max_rec = rec

    return max_rec

def PlateLocation(img, plate_color):
    """
    Input: one img and the color of the plate

    Return: location: ((x, y), (width, height), angle)
    """
    img_processed = preprocess(img, plate_color)
    if debug:
        show_img(img_processed)

    location = get_plate_location(img, img_processed)
    return location

if __name__=='__main__':
    # for img_path in ["test3.jpg"]:
    for img_path,color in [("test1.jpg", "blue"), ("test2.jpg","green"), ("test3.jpg","blue")]:
        print('--------- test %s -------'%img_path)
        img = cv2.imread(img_path)

        location  = PlateLocation(img, plate_color=color)
        draw_box(img, location, "%s_result"%img_path)

        print('The location of the plate:')
        print(' position x and y:', location[0][0],location[0][1])
        print(' width and height:', location[1][0], location[1][1])
        print(' rotation angle:', location[2])

    # cv2.imwrite("result.jpg", img)
