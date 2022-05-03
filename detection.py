# Candidate small object detection: Motion-Based Detection Using Local Noise Modelling Algorithm 

#import relevant packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#just test algorithm with three images 
img1 = cv2.imread("./mot/car/001/img/000001.jpg")
img2 = cv2.imread("./mot/car/001/img/000002.jpg")
img3 = cv2.imread("./mot/car/001/img/000003.jpg")

#Convert from BGR to Greyscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

#(1) Inter-frame differences
diff_img_12 = cv2.absdiff(gray2, gray1)
diff_img_23 = cv2.absdiff(gray3, gray2)

#(2) Thresholding: convert greyscale to binary 
pfa = 0.05
th_12 = - np.log(pfa) * np.mean(diff_img_12)
th_23 = - np.log(pfa) * np.mean(diff_img_23)
thresh_img_12 = cv2.threshold(diff_img_12, th_12, 255, cv2.THRESH_BINARY)[1]
thresh_img_23 = cv2.threshold(diff_img_23, th_23, 255, cv2.THRESH_BINARY)[1]

plt.imshow(thresh_img_12, cmap='gray')
plt.title("Thresholded Image: 12")
plt.show()

#(3) Candidate Extractions: Logical AND 
thresh_and = np.logical_and(thresh_img_12,thresh_img_23) * 255

plt.imshow(thresh_and, cmap='gray')
plt.title("logical AND")
plt.show()

### To Resolve ###
# local tactic: split into 30 x 30 regions
