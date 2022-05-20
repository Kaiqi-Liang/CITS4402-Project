import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2

def kalman(hypothesis):
    #the length of hypothesis is the number of frames. Each list inside hypothesis represents one frame's information including the image and the centroid 

    #initialize tracks on first frame 
    initializedTracks = init_tracks(hypothesis)
    frame_num =  1

    print(len(hypothesis))
    kalmanOutput = []
    kalmanOutput.append(initializedTracks)
    while frame_num < len(hypothesis):
        #currently looping over 0 and 1 
        #loop over the frames, excluding the last

        predictedTracks = predict_tracks(initializedTracks)
        matchedTracks = track_association(predictedTracks, hypothesis[frame_num][1])
        updatedTracks = updated_tracks(matchedTracks)
        kalmanOutput.append(updatedTracks)

        #reassigned initializedTracks to updatedTracks to pass onwards 
        initializedTracks = updatedTracks
        frame_num += 1

    for i, frame in enumerate(kalmanOutput):
        for cluster in frame:

            x_centroid = round(cluster[1][0,0])
            y_centroid = round(cluster[1][1,0])

            cv2.rectangle(hypothesis[i][0], (x_centroid - 5, y_centroid - 5), (x_centroid + 5, y_centroid + 5), (255, 0, 0), 2)
    
    plt.subplot(1,3,1)
    plt.imshow(hypothesis[0][0])
    plt.title("Frame 1")

    plt.subplot(1,3,2)
    plt.imshow(hypothesis[1][0])
    plt.title("Frame 2")

    plt.subplot(1,3,3)
    plt.imshow(hypothesis[2][0])
    plt.title("Frame 3")
    plt.show()

    video = cv2.VideoWriter('video.avi', 0, 1, (1025,1025))
    for frame in hypothesis:
        image = frame[0]
        video.write(image)
    
    cv2.destroyAllWindows()
    video.release()


    return


def init_tracks(hypothesis):
    '''
	Input: for each frame index n from 1 to N-1, this step takes as input a list of clusters, each cluster containing centroid and bounding box information
	Output: for each frame index n from 1 to N-1, this step outputs initalized cluster information containig an initialized state vector and track ID. This will be passed into 'predict' 
	'''
    #initialise tracks using the first frame. hypothesis[0] gives you the first frame. hypothesis[0][1] gives you the first frames clusters 
    max_ID = 0
    initializedTracks = []
    for cluster in hypothesis[0][1]:

        track_ID = max_ID
        state = np.matrix([cluster[0], cluster[1], 0, 0, 0, 0]).T
        cov = np.diag([1] * 6)
        max_ID += 1
        initializedTracks.append((track_ID, state, cov))

    # print(initializedTracks)    

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

    predicted_tracks = []

    for track in initializedTracks:
        track_ID, current_state, current_cov = track
        predicted_state = np.matmul(F, current_state)
        predicted_cov = np.matmul(F, np.matmul(current_cov, F.T)) + Q
        predicted_tracks.append((track_ID, predicted_state, predicted_cov))

    return predicted_tracks


def track_association(predicted_tracks, hypothesis):
    '''
    To be run after the predicted step. Goal: to match each track to the most plausible hypotheses 
	Input: for each frame index n from 1 to N-1, this step takes as input tuple of track ID and *****initialized***** state vector for each candidate cluster 
	Output: for each frame index n from 1 to N-1, this step outputs tuple of track ID and *****predicted***** state vector for each candidate cluster 
	'''

    predicted_clusters = predicted_tracks
    measured_clusters = hypothesis
    length = len(predicted_clusters) + len(measured_clusters)

    predicted_centroids = [(predicted_state[0, 0], predicted_state[1, 0]) for _, predicted_state, _ in predicted_clusters]
    measured_centroids = measured_clusters

    cost_matrix = np.zeros((length, length))
    for i, predicted_centroid in enumerate(predicted_centroids):
        for j, measured_centroid in enumerate(measured_centroids):
            cost_matrix[i][j] = sp.spatial.distance.euclidean(predicted_centroid, measured_centroid)
    mean = cost_matrix[:len(predicted_clusters), :len(measured_clusters)].mean()
    cost_matrix[len(predicted_clusters):, :len(measured_clusters)] = mean
    cost_matrix[:len(predicted_clusters), len(measured_clusters):] = mean
 
    predicted_idx, measured_idx = sp.optimize.linear_sum_assignment(cost_matrix)

    # print(predicted_idx, measured_idx)
    # return

    #Unassigned tracks occur when predicted values are assigned to pseudo hypothesis 
    #if there are n predicted values and m hypothesis 
    #then unassigned tracks occur when predicted[:n] is assigned to hypothesis[m:]
    matched_pairs =[]
    unassigned_tracks =[]
    unassigned_hypothesis = []
    for idx in range(length):
        proposed_pair = (predicted_idx[idx],measured_idx[idx])

        if proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):            
            matched_pairs.append((predicted_clusters[proposed_pair[0]],measured_centroids[proposed_pair[1]]))
            print("track_ID that was matched forward",predicted_clusters[proposed_pair[0]][0])

        elif proposed_pair[0] < len(predicted_clusters) and proposed_pair[1] >= len(measured_clusters):
            #matching predicted tracks to pseudo hypothesis
            #i.e. unassigned tracks
            print("track_ID that was unassigned",predicted_clusters[proposed_pair[0]][0] )

            unassigned_tracks.append(predicted_centroids[proposed_pair[0]])
            
        elif proposed_pair[0] >= len(predicted_clusters) and proposed_pair[1] < len(measured_clusters):
            #matching pseudo tracks to real hypothesis 
            #unassigned hypothesis 
            print("hypothesis that was unassigned",proposed_pair[1] )

            unassigned_hypothesis.append(measured_centroids[proposed_pair[1]])
            
        elif proposed_pair[0] >= len(predicted_clusters) and  proposed_pair[1] >= len(measured_clusters):
            #matching pseudo tracks to pseudo hypothesis 
            #i.e.
            pass

    # init_tracks(unassigned_hypothesis)
    # nearest_search(unassigned_tracks)

    return matched_pairs 

def updated_tracks(matched_pairs):
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
        track_ID = track[0]
        predicted_state = track[1]
        predicted_cov = track[2]
        measured_centroid = np.matrix(measured_centroids[i]).T

        innovation = measured_centroid - np.matmul(H,predicted_state)
        innovation_cov = np.matmul(np.matmul(H, predicted_cov),H.T) + R
        kalman_gain = np.matmul(np.matmul(predicted_cov, H.T), np.linalg.inv(innovation_cov))
        updated_state = predicted_state + np.matmul(kalman_gain,innovation)
        updated_cov = np.matmul((np.identity(6) - np.matmul(kalman_gain,H)),predicted_cov)

        updated_tracks.append((track_ID, updated_state, updated_cov))
        
    return updated_tracks


# for unassigned track ID's, i.e. gt clusters with no match in hypothesis     
def nearest_search(unassigned_tracks, frame, previous_frame):
    res = []
    # iterate through all unassigned tracks
    for gt_centroid in unassigned_tracks:
        # get grey image of previous frame
        _, grey_image2, _ = previous_frame
        # get grey image of current frame
        _, grey_image1, _ = frame
        # get centroid coordinates of unassigned track in current frame grey image
        row, column = gt_centroid
        # do template matching between unassigned track in current frame grey image with an area around unassigned track(im_area) in the previous frame grey image
        # get box area coordinates of unassigned track in current frame grey image which is the template for template matching
        template = grey_image1[row - 2 : row + 2, column - 2 : column + 2]
        # get box area coordinates of image area to search in the previous frame grey image to search for the template
        im_area = grey_image2[row - 3 : row + 3, column - 3 : column + 3]
        h, w = template.shape[::]
        res = cv2.matchTemplate(im_area, template, cv2.TM_SQDIFF) 
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
        # min_val is calculated assuming a 70% template match to confirm the template was found in the image area
        # min_val = (4 x 4 x 255**2) * 0.30 = approx. 300000 , the lower min_val is the better is the match 
        # if the template match was found in the image area of the previous frame, then get the hypothesis_centroid coordinates which
        # means the object did not move and the hypothesis object was not picked up in the current frame
        if min_val < 300000:
            # get top left coordinates of hypothesis object
            top_left = min_loc
            # calculate the hypothesis_centroid coordinates
            hypothesis_centroid = (top_left[0] + w/2, top_left[1] + h/2) 
        # append the hypothesis_centroid coordinates with the unassigned track coordinates which are matching objects
        res.append((hypothesis_centroid, gt_centroid))
    return res