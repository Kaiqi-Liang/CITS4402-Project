'''
Candidate small object detection: Motion-Based Detection Using Local Noise Modelling Algorithm
'''
import matplotlib.pyplot as plt
import numpy as np
from parser import Parser
import itertools
import cv2

def candidate_small_objects_detection(parser: Parser) -> list[np.ndarray]:
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input the frames at index n-1, n and n+1
	Output: for each frame index n from 1 to N-1, this step outputs a binary image representing candidate small objects
	'''
	start, end = parser.get_frame_range()
	output = []
	for frame in range(start + 1, end - 1):
		frame_behind = parser.load_frame(frame - 1)
		frame_center = parser.load_frame(frame)
		frame_front = parser.load_frame(frame + 1)

		#Convert from BGR to Greyscale
		gray_behind = cv2.cvtColor(frame_behind, cv2.COLOR_BGR2GRAY)
		gray_center = cv2.cvtColor(frame_center, cv2.COLOR_BGR2GRAY)
		gray_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)

		binary_rows = []
		for i in range(len(gray_behind) // 30 + 1):
			binary_cols = []
			for j in range(len(gray_behind) // 30 + 1):
				gray1_split = gray_behind[30 * i:30 * (i + 1), 30 * j:30 * (j + 1)]
				gray2_split = gray_center[30 * i:30 * (i + 1), 30 * j:30 * (j + 1)]
				gray3_split = gray_front[30 * i:30 * (i + 1), 30 * j:30 * (j + 1)]

				#(1) Inter-frame differences
				diff_img_12 = cv2.absdiff(gray2_split, gray1_split)
				diff_img_23 = cv2.absdiff(gray3_split, gray2_split)

				#(2) Threshold: convert greyscale to binary
				PFA = 0.05
				th_12 = -np.log(PFA) * np.mean(diff_img_12)
				th_23 = -np.log(PFA) * np.mean(diff_img_23)
				thresh_img_12 = cv2.threshold(diff_img_12, th_12, 1, cv2.THRESH_BINARY)[1]
				thresh_img_23 = cv2.threshold(diff_img_23, th_23, 1, cv2.THRESH_BINARY)[1]

				#(3) Candidate Extractions: Logical AND
				binary_cols.append(np.logical_and(thresh_img_12, thresh_img_23))

			binary_rows.append(binary_cols)

		# merge the 30x30 split images back into 1 binary image
		binary_image = np.concatenate([[list(itertools.chain(*col)) for col in zip(*row)] for row in binary_rows])

		# for each frame append the binary image, the center gray image and the frame number
		output.append((frame_center, gray_center, binary_image, frame))

	# plt.title('candidate small object detection')
	# plt.imshow(output[0][0], 'gray')
	# plt.savefig('candidate_detection.jpg')
	return output