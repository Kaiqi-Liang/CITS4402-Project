import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 
import skimage
import cv2

def candidate_match_discrimination(frames: list[np.ndarray], areaTh: tuple[float, float], extendTh: tuple[float, float], majorAxisTh: tuple[float, float], eccentricityTh: tuple[float, float]):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a binary image representing the candidate small objects.
	Output: for each frame index n from 1 to N-1, this step outputs the bounding box and centroid of each candidate small object.
	'''
	for _, gray_image, binary_image, _ in frames:
		#(1) Region Growing: for each frame in frames. Each frame has a binary image and a greyscale image 

		# label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)
		clusters = skimage.measure.regionprops(labelled_image)
		for cluster in clusters:

			# centroid of cluster 
			row, col = cluster.centroid
			row = round(row)
			col = round(col)

			# 11 x 11 search window centered around centroid
			binary_window = binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]
			gray_window = gray_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]

			gray_binary = gray_window[binary_window]

			# the objects being detected are quite small, so we set a higher thresholds that any regions that are smaller than 2 pixels are ruled out
			if len(gray_binary) <= 2:
				continue

			# find the pixel mean of the window
			window_mean = np.average(gray_binary)
			# find standard deviation of the window
			window_std = np.std(gray_binary)

			# get the upper quantile limit of candidate cluster
			upper_th = min(sp.stats.norm.ppf(0.995, loc=window_mean, scale=window_std), 255)
			# get the lower quantile limit of candidate cluster
			lower_th = sp.stats.norm.ppf(0.005, loc=window_mean, scale=window_std)			

			# mark the pixels within the quantile interval as being part of candidate cluster in object

			for i, pixels in enumerate(gray_window):
				for j, pixel in enumerate(pixels):
					if lower_th <= pixel <= upper_th:
						binary_window[i,j] = True
			binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)] = binary_window

	# plt.title('region growing')
	# plt.imshow(frames[0][2], 'gray')
	# plt.savefig('region_growing.jpg')
	# plt.show()

	#(2) Morphological Cues
	output = [] #list of candidate small objects for each frame 

	# Threshold values
	area_lower, area_upper = areaTh
	extent_lower, extent_upper = extendTh
	major_axis_lower, major_axis_upper = majorAxisTh
	eccentricity_lower, eccentricity_upper = eccentricityTh

	for original_image, _, binary_image, frame in frames:
		image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		candidate_small_objects = [] #candidate small objects in each frame
		# label connected regions in now 'grown' binary image
		labelled_image = skimage.measure.label(binary_image)

		clusters = skimage.measure.regionprops(labelled_image)
		for cluster in clusters:
			area = cluster.area
			extent = area / cluster.area_bbox
			major_axis = cluster.axis_major_length
			eccentricity = cluster.eccentricity

			if area_lower <= area <= area_upper and extent_lower <= extent <= extent_upper and major_axis_lower <= major_axis <= major_axis_upper and eccentricity_lower <= eccentricity <= eccentricity_upper:
				candidate_small_objects.append((cluster.centroid, cluster.bbox))
				min_row, min_col, max_row, max_col = cluster.bbox
				original_image
				cv2.rectangle(image, (min_row, min_col), (max_row, max_col), (255, 0, 0), 2)
		print(frame)
		plt.title(f'frame {frame}')
		plt.imshow(image)
		# plt.savefig(f'{frame}.jpg')
		plt.show()
		output.append(candidate_small_objects)

	return output












		# intersection over union
		# for gt_box in gt:
		# 	topleftx, toplefty, width, height = gt_box[-4:]
		# 	min_row = toplefty
		# 	min_col = topleftx
		# 	max_row = min_row + height
		# 	max_col = min_col + width
		# 	gt_res.append((min_row, min_col, max_row, max_col))

	# max_track_id = gt[-1][0]
	# match = False
	# for hypothesis in pred_res:
	# 	for gt_track in gt_res:
	# 		if (bb_intersection_over_union(hypothesis[0], gt_track) >= 0.7):
	# 			print(hypothesis[0], gt_track)
	# 			match = True
	# 	if not match:
	# 		max_track_id += 1
	# 		filter_init(hypothesis, max_track_id)

def bb_intersection_over_union(box_pred: list[int], box_gt: list[int]):
	#determine x-y coords of the intersection rectangle 
	xA = max(box_pred[0], box_gt[0]) # min row 
	yA = max(box_pred[1], box_gt[1]) # min col
	xB = min(box_pred[2], box_gt[2]) # max row
	yB = min(box_pred[3], box_gt[3]) # max col 

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0

 	# compute the area of both the prediction and ground-truth
    # rectangles
	box_predArea = abs((box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1]))	
	box_gtArea = abs((box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1]))
	
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
	iou = interArea / float(box_predArea + box_gtArea - interArea)

	return iou

# calculate measure of quality of predicted candidate clusters
def accuracy(pred_res: list[np.ndarray], gt_res: list[np.ndarray]):
	pred_res1 = []
	gt_res1 = []
	true_positive = 0
	false_positive = 0
	false_negative = 0
	for binary_image, centroid in pred_res:
		for box_gt in gt_res:
			cal_iou = bb_intersection_over_union(binary_image, box_gt)
			# if true positive append the binary image
			if cal_iou >= 0.7:
				true_positive += 1
				pred_res1.append([binary_image, centroid])
				gt_res1.append([box_gt])
			# if false positive
			elif binary_image and cal_iou <= 0.7 :
				false_positive += 1
			# if false negative
			elif box_gt and cal_iou <= 0.7 :
				false_negative += 1
		if len(pred_res) == 5 :
			break
		print(pred_res)
		print(gt_res)
	# calculate precision and recall
	precision = true_positive / (true_positive + false_positive)
	print(precision)
	recall = true_positive / (true_positive + false_negative)		
	print(recall)
	F1 = 2 * ((precision*recall)/(precision+recall))
	print(F1)
	return pred_res	