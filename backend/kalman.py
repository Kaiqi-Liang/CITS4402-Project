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

def init_tracks(unassigned_hypothesis):
    #for unassigned hypothesis, i.e. predicted clusters with no match in gt

    # state = pd.DataFrame[centroid[0], centroid[1], 0, 0, 0, 0]

    # std = np.arra
    # motion_cov = np.diag(std)
    pass
