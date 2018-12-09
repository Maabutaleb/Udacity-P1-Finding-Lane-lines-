#This code reads videos and display
#To run it , you need to change video path at the code below 
import numpy as np
import cv2
import math
import os


#Intitating Parameters 

Quad_bottom_width = 0.85 
Quad_top_width = 0.07 
Quad_height = 0.4 

rho = 5                  # distance resolution in pixels of the Hough grid
theta = np.pi/180        # angular resolution in radians of the Hough grid
threshold = 350          # minimum number of votes (intersections in Hough grid cell)
min_line_length = 0.1    # minimum number of pixels making up a line
max_line_gap = 1         # maximum gap in pixels between connectable line segments


#Defining Functions 

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def grayscale(img):
    #img=cv2.imread(img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def weighted_img(img, initial_img,a=0.8, b=1., s=0.):
    return cv2.addWeighted(initial_img, a, img, b, s)


def filter_colors(image):
    white_threshold = 200 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2

def Disp_Save_Result(img):
    
        cv2.imshow('Result',img)
        
        k = cv2.waitKey(0)
        
        if k == 27:

           cv2.destroyAllWindows()

        elif k == ord('s'):
            #Note : This line of code reads videos from path on my pc ,
            #I could not make a code to read directly from a folder at the
            #Same directory 
            cv2.imwrite('C:\\Users\\MAHMOUD-DELL-LAPTOP\\Desktop\\My Career Dev. Prog. - Embedded & Auton. Driving\\Self Driving Cars Nano Degree\\Term 1\\P1\\Trials and Testings\\Project Building and Testing\\test_images_output\\solidYellowCurve2_r2.jpg',img)
            #cv2.imwrite('result.jpg',img)
            cv2.destroyAllWindows()                       

#######################################################################
###################Program Starts from here############################
#######################################################################
            
#Read Video
cap = cv2.VideoCapture("solidWhiteRight.mp4")
out = cv2.VideoWriter('output2.mp4', -1, 20.0, (640,480))

#out=cv2.Videowriter('solidWhiteRightr',cv.CV_FOURCC('M','J','P','G'),32,(640,360),1)

# Define the codec and create VideoWriter object
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        img = filter_colors(frame)
        img = grayscale(img)
        img = canny(img,50,150)
        img = gaussian_blur(img,5)
        # Apply ROI Quad polygon mask
        imshape = img.shape
        vertices = np.array([[\
        ((imshape[1] * (1 - Quad_bottom_width)) // 2, imshape[0]),\
        ((imshape[1] * (1 - Quad_top_width)) // 2, imshape[0] - imshape[0] * Quad_height),\
        (imshape[1] - (imshape[1] * (1 - Quad_top_width)) // 2, imshape[0] - imshape[0] * Quad_height),\
        (imshape[1] - (imshape[1] * (1 - Quad_bottom_width)) // 2, imshape[0])]]\
        ,dtype=np.int32)
        img = region_of_interest(img , vertices)
        img = hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
        initial_image = frame.astype('uint8')
        img = weighted_img(img,initial_image)

        
        # write the flipped frame
        cv2.imshow("Result", img)
        out.write(img)
        # cv2.imwrite('C:\\Users\\MAHMOUD-DELL-LAPTOP\\Desktop\\My Career Dev. Prog. - Embedded & Auton. Driving\\Self Driving Cars Nano Degree\\Term 1\\P1\\Trials and Testings\\solidWhiteRight_r.mp4',img)
        key = cv2.waitKey(20)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()


             
    

