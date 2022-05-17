import numpy as np
import scipy as sp

def track_association(frames):

    for hypotheses, gt in frames:
        length = max(len(hypotheses), len(gt))
        gt_centroid = [(centroid[1], centroid[2]) for centroid in gt]
        hypotheses_centroid = [centroid for centroid, _ in hypotheses]

        cost_matrix = np.zeros((length, length))
        cost_matrix.fill(10)
        for i, hypothesis_centroid in enumerate(hypotheses_centroid):
            for j, centroid in enumerate(gt_centroid):
                cost_matrix[i][j] = sp.spatial.distance.euclidean(hypothesis_centroid, centroid)

        hypotheses_idx, gt_idx = sp.optimize.linear_sum_assignment(cost_matrix)

        # print(hypotheses_idx, gt_idx)

        if len(hypotheses) > len(gt):
            # more hypothesis centroids then gt centroids
            # thus there will be unassigned hypothesis - these will be passed into filter_init
            diff = length - len(gt_centroid)
            matched_clusters = []
            for idx in range(len(gt)):
                matched_clusters.append([hypotheses_centroid[list(gt_idx).index(idx)], gt_centroid[idx]])
            
            # hypothesis assigned to a pseudo track are unassigned hypothesis
            unassigned_hypothesis = []
            for idx in hypotheses_idx[-diff:]:
                unassigned_hypothesis.append(hypotheses_centroid[list(gt_idx).index(idx)])

            init_tracks(unassigned_hypothesis)
            # print(unassigned_hypothesis)
            # print(matched_clusters)

        elif len(hypotheses) < len(gt):
            # more gt centroids than hypothesis centroids
            # thus there will be unassigned track ID's - these will be passed in nearest search 
            diff = length - len(hypotheses)
            matched_clusters = []
            for idx in range(len(hypotheses)):
                matched_clusters.append([hypotheses_centroid[idx], gt_centroid[gt_idx[idx]]])

            # tracks assigned to pseudo hypothesis are unassigned tracks 
            unassigned_tracks = []
            for idx in hypotheses_idx[-diff:]:
                unassigned_tracks.append(gt_centroid[list(gt_idx).index(idx)])



            # print(unassigned_tracks)
            # print(matched_clusters)
        
        else:
            # same amount of clusters in both 
            matched_clusters = []
            for i in range(len(hypotheses)):
                matched_clusters.append([hypotheses_centroid[i], gt_centroid[gt_idx[i]]])
    
    return matched_clusters 

#for unassigned track ID's, i.e. gt clusters with no match in hypothesis     
def nearest_search(unassigned_tracks, previous_frame):
    for hypotheses,_ in previous_frame:
        hypotheses_centroid = [centroid for centroid, _ in hypotheses]

    length = len(hypotheses_centroid)
    cost_matrix = np.zeros((length, length))
    cost_matrix.fill(70)

    for i, hypothesis_centroid in enumerate(hypotheses_centroid):
        for j, centroid in enumerate(unassigned_tracks):
            cost_matrix[i][j] = sp.spatial.distance.euclidean(hypothesis_centroid, centroid)

    hypotheses_idx, gt_idx = sp.optimize.linear_sum_assignment(cost_matrix)            

    candidate_clusters = []
    for idx in range(len(unassigned_tracks)):
        candidate_clusters.append([hypotheses_centroid[list(gt_idx).index(idx)], unassigned_tracks[idx]])

    matched_clusters = []
    # Do template matching for candidate clusters, if template matching is high then confirm matched clusters and 
    # append to matched_clusters
    
    return matched_clusters

def init_tracks(frames):
    max_ID = 0
    output = []
    for frame in frames:
        tracks = []
        for cluster in frame:
            track_ID = max_ID
            state = np.array([cluster[0][0], cluster[0][1], 0, 0, 0, 0])
            cov = np.diag([1] * 6)
            cluster_info = (track_ID, state, cov)
            max_ID += 1
            tracks.append(cluster_info)
        output.append((frame, tracks))
    return output

def predict(tracks):
    tstep = 1
    F = np.diag([1.0] * 6)
    np.fill_diagonal(F[:-2,2:], tstep)
    np.fill_diagonal(F[:-4,4:], (tstep ** 2) / 2)
    F = np.matrix(F)
    Q = np.matrix(np.diag([1] * 6))
    predicted_state = []
    for track in tracks:
        current_state = track[1]
        P = track[2]
        future_state = np.matmul(F, current_state.T)
        future_cov = np.matmul(F, np.matmul(P, F.T)) + Q
        predicted_state.append((future_state, future_cov))
    return predicted_state