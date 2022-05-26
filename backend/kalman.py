import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2

def kalman(hypothesis, non_assignment_cost = None):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a list of clusters, each cluster containing centroid and bounding box information
	Output: for each frame index n from 1 to N-1, this step outputs initialised cluster information containing an initialized state vector and track ID. This will be passed into 'predict'
	'''
	# Initialize tracks on using the first frame frame 
	initializedTracks = init_tracks(hypothesis)
	frame_num = 1

	kalmanOutput = [initializedTracks]
	while frame_num < len(hypothesis):
		# Loop over the frames, excluding the last
		predictedTracks = predict_tracks(initializedTracks)
		matchedTracks = track_association(predictedTracks, hypothesis[frame_num][1], non_assignment_cost)
		updatedTracks = updated_tracks(matchedTracks)
		kalmanOutput.append(updatedTracks)

		# Reassigned initializedTracks to updatedTracks to pass onwards 
		initializedTracks = updatedTracks
		frame_num += 1

	for i, frame in enumerate(kalmanOutput):
		for cluster in frame:
			x_centroid = round(cluster[1][0,0])
			y_centroid = round(cluster[1][1,0])
			cv2.rectangle(hypothesis[i][0], (x_centroid - 5, y_centroid - 5), (x_centroid + 5, y_centroid + 5), (255, 0, 0), 2)
		plt.title(f'frame {hypothesis[i][2]}')
		plt.imshow(hypothesis[i][0])
		plt.savefig(f'{hypothesis[i][2]}.jpg')

	plt.figure()
	plt.xlabel('Frame')
	plt.ylabel('Number of moving objects detected')
	plt.plot([frame[-1] for frame in hypothesis], [len(frame) for frame in kalmanOutput])
	plt.savefig('graph.jpg')

def init_tracks(hypothesis):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a list of clusters, each cluster containing centroid and bounding box information
	Output: for each frame index n from 1 to N-1, this step outputs initialized cluster information containing an initialized state vector and track ID. This will be passed into 'predict'
	'''
	# Initialise tracks using the first frame
	max_ID = 0
	initializedTracks = []
	for cluster in hypothesis[0][1]:
		track_ID = max_ID
		state = np.matrix([cluster[0], cluster[1], 0, 0, 0, 0]).T
		Q = np.matrix(np.diag([1] * 6))
		for i in range(2,6):
			Q[i,i] = 4
		max_ID += 1
		initializedTracks.append((track_ID, state, Q))
	return initializedTracks

def predict_tracks(initializedTracks):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''
	# Time step between frames
	tstep = 1

	# F matrix 
	F = np.diag([1.0] * 6)
	np.fill_diagonal(F[:-2,2:], tstep)
	np.fill_diagonal(F[:-4,4:], (tstep ** 2) / 2)
	F = np.matrix(F)

	# Q matrix 
	Q = np.matrix(np.diag([1] * 6))
	for i in range(2,6):
		Q[i,i] = 4

	# Kalman filter prediction step for tracks
	predicted_tracks = []
	for track in initializedTracks:
		track_ID, current_state, current_cov = track
		predicted_state = np.matmul(F, current_state)
		predicted_cov = np.matmul(F, np.matmul(current_cov, F.T)) + Q
		predicted_tracks.append((track_ID, predicted_state, predicted_cov))
	return predicted_tracks

def track_association(predicted_tracks, hypothesis, non_assignment_cost):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''

	predicted_clusters = predicted_tracks
	measured_clusters = hypothesis

	predicted_centroids = [(predicted_state[0, 0], predicted_state[1, 0]) for _, predicted_state, _ in predicted_clusters]
	measured_centroids = measured_clusters

	# The cost matrix square and size is equal to the sum of tracks and hypothesis
	length = len(predicted_clusters) + len(measured_clusters)

	# Initialize cost matrix as zero. Fill real matches using euclidean distance and fill pseudo matches using non_assignment_cost constant
	cost_matrix = np.zeros((length, length))

	for i, predicted_centroid in enumerate(predicted_centroids):
		for j, measured_centroid in enumerate(measured_centroids):
			cost_matrix[i][j] = sp.spatial.distance.euclidean(predicted_centroid, measured_centroid)

	# If non_assignment_cost is not provided, default to slightly larger than the mean of euclidean distances. The cost of non assignment should be high so to avoid this scenario
	mean = cost_matrix[:len(predicted_clusters), :len(measured_clusters)].mean() + 10
	if non_assignment_cost is None:
		non_assignment_cost = mean
	cost_matrix[len(predicted_clusters):, :len(measured_clusters)] = non_assignment_cost
	cost_matrix[:len(predicted_clusters), len(measured_clusters):] = non_assignment_cost

	# Optimise the assignment of tracks and hypothesis
	predicted_idx, measured_idx = sp.optimize.linear_sum_assignment(cost_matrix)

	matched_pairs = []
	unassigned_tracks = []
	unassigned_hypothesis = []

	# Loop over every proposed matching pair and assign the pair as every a 'matched_pair', 'unassigned_track', or 'unassigned_hypothesis'
	for idx in range(length):
		proposed_pair = (predicted_idx[idx],measured_idx[idx])

		if proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):      
			#matching predicted tracks to measured hypothesis
			matched_pairs.append((predicted_clusters[proposed_pair[0]],measured_centroids[proposed_pair[1]]))

		elif proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] >= len(measured_clusters):
			#matching predicted tracks to pseudo hypothesis
			unassigned_tracks.append([proposed_pair[0],predicted_centroids[proposed_pair[0]]])
			
			#template_matching(unassigned_tracks)
			#the template_matching function should be run on unassigned tracks, however the integration of this function within the Kalman cycle was not completed 
			
		elif proposed_pair[0] >= len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):
			#matching pseudo tracks to real hypothesis - i.e. unassigned hypothesis 
			unassigned_hypothesis.append(measured_centroids[proposed_pair[1]])

			#init_tracks(unassigned_hypothesis)
			#the init_tracks function should be run on unassigned_hypothesis, however the integration of this function within the Kalman cycle was not completed 

	return matched_pairs 

def updated_tracks(matched_pairs):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''
	# Define constants
	H = np.zeros((2, 6))
	np.fill_diagonal(H[:2,:2],1)
	R = np.diag([1.0] * 2)

	predicted_tracks = []
	measured_centroids = []
	updated_tracks = []

	# Loop over every matched track and hypothesis
	for pair in matched_pairs:
		predicted_tracks.append(pair[0])
		measured_centroids.append(pair[1])

	# Loop over every predicted track and perform kalman update using measured information, i.e. matched hypothesis
	for i, track in enumerate(predicted_tracks):
		track_ID = track[0]
		predicted_state = track[1]
		predicted_cov = track[2]
		measured_centroid = np.matrix(measured_centroids[i]).T

		# Kalman filter update step for tracks
		innovation = measured_centroid - np.matmul(H,predicted_state)
		innovation_cov = np.matmul(np.matmul(H, predicted_cov),H.T) + R
		kalman_gain = np.matmul(np.matmul(predicted_cov, H.T), np.linalg.inv(innovation_cov))
		updated_state = predicted_state + np.matmul(kalman_gain,innovation)
		updated_cov = np.matmul((np.identity(6) - np.matmul(kalman_gain,H)),predicted_cov)

		updated_tracks.append((track_ID, updated_state, updated_cov))
	return updated_tracks

def template_matching(unassigned_tracks, frame, previous_frame):
	'''
	Not complete.
	Input: list of unassigned tracks, including track ID, centroid, current frame and previous frame. 
	Output: hypothesis centroid and track centroid in a list 
	'''
	res = []

	# iterate through all unassigned tracks
	for track in unassigned_tracks:

		# get grey image of previous frame
		_, grey_previous, _ = previous_frame

		# get grey image of current frame
		_, grey_current, _ = frame

		# get centroid coordinates of unassigned track in current frame grey image
		row, column = track

		# do template matching between unassigned track in current frame grey image with an area around unassigned track(im_area) in the previous frame grey image
		# get box area coordinates of unassigned track in current frame grey image which is the template for template matching
		template = grey_current[row - 3 : row + 3, column - 3 : column + 3]
		# get box area coordinates of image area to search in the previous frame grey image to search for the template
		search_area = grey_previous[row - 4 : row + 4, column - 4 : column + 4]
		h, w = template.shape[::]
		res = cv2.matchTemplate(search_area, template, cv2.TM_SQDIFF) 

		min_val, _, min_loc, _ = cv2.minMaxLoc(res)

		# min_val is calculated assuming a 70% template match to confirm the template was found in the image area
		# min_val = ( 6 x 6 x 255**2) * 0.30 = approx. 300000 , the lower min_val is the better is the match 
		# if the template match was found in the image area of the previous frame, then get the hypothesis_centroid coordinates which
		# means the object did not move and the hypothesis object was not picked up in the current frame
		if min_val < 700000:
			# get top left coordinates of hypothesis object
			top_left = min_loc
			# calculate the hypothesis_centroid coordinates
			hypothesis_centroid = (top_left[0] + w/2, top_left[1] + h/2) 
		# append the hypothesis_centroid coordinates with the unassigned track coordinates which are matching objects
		res.append((hypothesis_centroid, track))

	return res