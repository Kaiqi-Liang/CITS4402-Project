import numpy as np
import scipy as sp

def kalman():
    pass

def init_tracks(frameCandidateClusters):
    '''
	Input: for each frame index n from 1 to N-1, this step takes as input a list of clusters, each cluster containing centroid and bounding box information
	Output: for each frame index n from 1 to N-1, this step outputs initalized cluster information containig an initialized state vector and track ID. This will be passed into 'predict' 
	'''
    #initialise tracks using the first frame 
    max_ID = 0
    initializedtracks = []
    for cluster in frameCandidateClusters[0]:
        track_ID = max_ID
        state = np.array([cluster[0], cluster[1], 0, 0, 0, 0])
        cov = np.diag([1] * 6)
        max_ID += 1
        initializedtracks.append((track_ID, state, cov))

        #frame is all of the clusters in that frame 
        # tracks is initalized tracks for all the clusters in that frame 
    return initializedtracks

def predict(initializedtracks):
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

    predicted_tracks = []

    for track in initializedtracks:
        track_ID, current_state, current_cov = track
        predicted_state = np.matmul(F, current_state.T)
        predicted_cov = np.matmul(F, np.matmul(current_cov, F.T)) + Q
        predicted_tracks.append((track_ID, predicted_state, predicted_cov))

    return predicted_tracks


def track_association(predicted_tracks, frameCandidateClusters):
    '''
    To be run after the predicted step. Goal: to match each track to the most plausible hypotheses 
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''

    predicted_clusters = predicted_tracks
    measured_clusters = frameCandidateClusters[1]
    length = len(predicted_clusters) + len(measured_clusters)

    predicted_centroids = [(predicted_state[0, 0], predicted_state[0, 1]) for _, predicted_state, _ in predicted_clusters]
    measured_centroids = measured_clusters

    cost_matrix = np.zeros((length, length))
    for i, predicted_centroid in enumerate(predicted_centroids):
        for j, measured_centroid in enumerate(measured_centroids):
            cost_matrix[i][j] = sp.spatial.distance.euclidean(predicted_centroid, measured_centroid)
    mean = cost_matrix[:len(predicted_clusters), :len(measured_clusters)].mean()
    cost_matrix[len(predicted_clusters):, :len(measured_clusters)] = mean
    cost_matrix[:len(predicted_clusters), len(measured_clusters):] = mean
 
    predicted_idx, measured_idx = sp.optimize.linear_sum_assignment(cost_matrix)

    #Unassigned tracks occur when predicted values are assigned to pseudo hypothesis 
    #if there are n predicted values and m hypothesis 
    #then unassigned tracks occur when predicted[:n] is assigned to hypothesis[m:]
    matched_pairs =[]
    matched_predictions = []
    unassigned_tracks =[]
    unassigned_hypothesis = []
    for idx in range(length):
        proposed_pair = (predicted_idx[idx],measured_idx[idx])

        if proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):
            # matched_pairs.append((predicted_centroids[proposed_pair[0]], measured_centroids[proposed_pair[1]]))
            
            matched_pairs.append((predicted_clusters[proposed_pair[0]],measured_centroids[proposed_pair[1]]))
            matched_predictions.append((predicted_clusters[proposed_pair[0]]))

        elif proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] >= len(measured_clusters):
            #matching predicted tracks to pseudo hypothesis
            #i.e. unassigned tracks
            unassigned_tracks.append(predicted_centroids[proposed_pair[0]])
            
        elif proposed_pair[0] >= len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):
            #matching pseudo tracks to real hypothesis 
            #unassigned hypothesis 
            unassigned_hypothesis.append(measured_centroids[proposed_pair[1]])
            
        elif proposed_pair[0] >= len(predicted_clusters) and  proposed_pair[1] >= len(measured_clusters):
            #matching pseudo tracks to pseudo hypothesis 
            #i.e.
            pass

    # init_tracks(unassigned_hypothesis)
    # nearest_search(unassigned_tracks)

    return matched_pairs 

def updateKalman(matched_pairs):
    '''
    To be run after the tracks and hypothesis have been associated with one another. Updates the state estimate of the Kalman filter. 
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''

    # Define constants
    H = np.zeros((2, 6))
    np.fill_diagonal(H[:2,:2],1)
    R = np.diag([1.0] * 2)

    predicted_tracks = []
    measured_centroids = []

    for pair in matched_pairs:
        predicted_tracks.append(pair[0])
        measured_centroids.append(pair[1])

    updated_tracks = []
    for i, track in enumerate(predicted_tracks):
        predicted_state = track[1].T
        predicted_cov = track[2]
        measured_centroid = np.matrix(measured_centroids[i]).T
 
        innovation = measured_centroid - np.matmul(H,predicted_state)
        innovation_cov = np.matmul(np.matmul(H, predicted_cov),H.T) + R
        kalman_gain = np.matmul(np.matmul(predicted_cov, H.T), np.linalg.inv(innovation_cov))
        updated_state = predicted_state + np.matmul(kalman_gain,innovation)
        updated_cov = np.matmul((np.identity(6) - np.matmul(kalman_gain,H)),predicted_cov)

        updated_tracks.append((updated_state, updated_cov))

    return updated_tracks

# for unassigned track ID's, i.e. gt clusters with no match in hypothesis     
def nearest_search(unassigned_tracks, frame, previous_frame):
    res = []
    for gt_centroid in unassigned_tracks:
        _, grey_image2, _ = previous_frame
        _, grey_image1, _ = frame
        row, column = gt_centroid
        template = grey_image1[row - 2 : row + 2, column - 2 : column + 2]
        im_area = grey_image2[row - 3 : row + 3, column - 3 : column + 3]
        h, w = template.shape[::]
        res = cv2.matchTemplate(im_area, template, cv2.TM_SQDIFF) 
        if res < 300000:
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)   
            # top_left = min_loc
            # bottom_right = (top_left[0] + w, top_left[1 + h]) 
            hypothesis_centroid = gt_centroid
        res.append((hypothesis_centroid, gt_centroid))
    return res