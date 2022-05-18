import numpy as np
import scipy as sp

# Tracking Phase

def init_tracks(frameCandiateClusters):
    '''
	Input: for each frame index n from 1 to N-1, this step takes as input a list of clusters, each cluster containing centroid and bounding box information
	Output: for each frame index n from 1 to N-1, this step outputs initalized cluster information containig an initialized state vector and track ID. This will be passed into 'predict' 
	'''
    tracksFrames = []
    for frame in frameCandiateClusters:
        max_ID = 0
        tracks = []
        for cluster in frame:
            track_ID = max_ID
            state = np.array([cluster[0][0], cluster[0][1], 0, 0, 0, 0])
            cov = np.diag([1] * 6)
            cluster_info = (track_ID, state, cov)
            max_ID += 1
            tracks.append(cluster_info)
        tracksFrames.append((frame, tracks))
        #frame is all of the clusters in that frame 
        # tracks is initalized tracks for all the clusters in that frame 
    return tracksFrames

def predict(tracksFrames):
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
    predictedtracksFrames = []
    for _, tracks in tracksFrames:
        # This loops over the frames
        predicted_tracks = []
        for track in tracks:
            # This loops over each cluster in each frame 
            current_state = track[1]
            current_cov = track[2]
            predicted_state = np.matmul(F, current_state.T)
            predicted_cov = np.matmul(F, np.matmul(current_cov, F.T)) + Q
            cluster_info = (track[0], predicted_state, predicted_cov)
            predicted_tracks.append(cluster_info)
        predictedtracksFrames.append((predicted_tracks))
        
    return predictedtracksFrames

def track_association(predictedtracksFrames, frameCandiateClusters):
    '''
    To be run after the predicted step. Goal: to match each track to the most plausible hypotheses 
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''

    for n in range(len(predictedtracksFrames)):
        predicted_clusters = predictedtracksFrames[n][1][0:1]
        measured_clusters = frameCandiateClusters[n+1]
        print(predicted_clusters, measured_clusters)
        return

    return 
    
    # for hypotheses, gt in frames:
    #     length = max(len(hypotheses), len(gt))
    #     gt_centroid = [(centroid[1], centroid[2]) for centroid in gt]
    #     hypotheses_centroid = [centroid for centroid, _ in hypotheses]

    #     cost_matrix = np.zeros((length, length))
    #     cost_matrix.fill(10)
    #     for i, hypothesis_centroid in enumerate(hypotheses_centroid):
    #         for j, centroid in enumerate(gt_centroid):
    #             cost_matrix[i][j] = sp.spatial.distance.euclidean(hypothesis_centroid, centroid)

    #     hypotheses_idx, gt_idx = sp.optimize.linear_sum_assignment(cost_matrix)

    #     # print(hypotheses_idx, gt_idx)

    #     if len(hypotheses) > len(gt):
    #         # more hypothesis centroids then gt centroids
    #         # thus there will be unassigned hypothesis - these will be passed into filter_init
    #         diff = length - len(gt_centroid)
    #         matched_clusters = []
    #         for idx in range(len(gt)):
    #             matched_clusters.append([hypotheses_centroid[list(gt_idx).index(idx)], gt_centroid[idx]])
            
    #         # hypothesis assigned to a pseudo track are unassigned hypothesis
    #         unassigned_hypothesis = []
    #         for idx in hypotheses_idx[-diff:]:
    #             unassigned_hypothesis.append(hypotheses_centroid[list(gt_idx).index(idx)])

    #         init_tracks(unassigned_hypothesis)
    #         # print(unassigned_hypothesis)
    #         # print(matched_clusters)

    #     elif len(hypotheses) < len(gt):
    #         # more gt centroids than hypothesis centroids
    #         # thus there will be unassigned track ID's - these will be passed in nearest search 
    #         diff = length - len(hypotheses)
    #         matched_clusters = []
    #         for idx in range(len(hypotheses)):
    #             matched_clusters.append([hypotheses_centroid[idx], gt_centroid[gt_idx[idx]]])

    #         # tracks assigned to pseudo hypothesis are unassigned tracks 
    #         unassigned_tracks = []
    #         for idx in hypotheses_idx[-diff:]:
    #             unassigned_tracks.append(gt_centroid[list(gt_idx).index(idx)])



    #         # print(unassigned_tracks)
    #         # print(matched_clusters)
        
    #     else:
    #         # same amount of clusters in both 
    #         matched_clusters = []
    #         for i in range(len(hypotheses)):
    #             matched_clusters.append([hypotheses_centroid[i], gt_centroid[gt_idx[i]]])
    
    return matched_clusters 

def updateKalman(predictions, measurements):
    '''
    To be run after the tracks and hypothesis have been associated with one another. Updates the state estimate of the Kalman filter. 
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''
    H = np.zeros((2, 6))
    np.fill_diagonal(H[:2,:2],1)
    R = np.diag([1.0] * 2)

    predicted_location = predictions[0]
    measured_location = measurements[0]
    innovation = measured_location - np.matmul(H,predicted_location)
    innovation_cov = np.matmul(np.matmul(H, predicted_cov),H.T) + R
    kalman_gain = np.matmul(np.matmul(predicted_cov, H.T), np.linalg(innovation_cov))
    updated_state_location = predicted_location + np.matmul(kalman_gain,innovation)
    updated_state_cov = np.matmul((np.identity(6)-np.matmul(kalman_gain,H)),predicted_cov)

    return
    
#for unassigned track ID's, i.e. gt clusters with no match in hypothesis     
# def nearest_search(unassigned_tracks, frame, previous_frame):
#     res = []
#     for gt_centroid in unassigned_tracks:
#         _, grey_image2, _ = previous_frame
#         _, grey_image1, _ = frame
#         row, column = gt_centroid
#         template = grey_image1[row - 2 : row + 2, column - 2 : column + 2]
#         im_area = grey_image2[row - 3 : row + 3, column - 3 : column + 3]
#         h, w = template.shape[::]
#         res = cv2.matchTemplate(gt_image, template, cv2.TM_SQDIFF) 
#         if res > 70:
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)   
#             top_left = min_loc
#             bottom_right = (top_left[0] + w, top_left[1 + h]) 
#             hypothesis_centroid = gt_centroid
#         res.append((hypothesis_centroid, gt_centroid))
#     return res

