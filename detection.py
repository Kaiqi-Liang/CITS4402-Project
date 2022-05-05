# Candidate small object detection: Motion-Based Detection Using Local Noise Modelling Algorithm 

#import relevant packages 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from parser import Parser

def candidate_small_objects_detection(parser: Parser) -> list[np.ndarray]:
	start, end = parser.get_frame_range()
	res = []
	for n in range(start + 1, end - 1):
		img1 = parser.load_frame(n - 1)
		img2 = parser.load_frame(n)
		img3 = parser.load_frame(n + 1)

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

		# plt.imshow(thresh_img_12, cmap='gray')
		# plt.title("Thresholded Image: 12")
		# plt.show()

		#(3) Candidate Extractions: Logical AND 
		thresh_and = np.logical_and(thresh_img_12, thresh_img_23) * 255
		res.append(thresh_and)

		# plt.imshow(thresh_and, cmap='gray')
		# plt.title("logical AND")
		# plt.show()

	return res