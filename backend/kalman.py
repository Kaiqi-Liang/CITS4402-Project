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

        # if len(hypotheses) > length:
        #     # more hypothesis then gt centroids
        #     diff = len(gt_centroid)

        if len(hypotheses) < length:
            # more gt centroids than hypothesis centroids
            diff = length - len(hypotheses)
            matched_clusters = []
            for i in range(len(hypotheses)):
                matched_clusters.append([hypotheses_centroid[i], gt_centroid[gt_idx[i]]])

            unassigned_tracks = []
            for idx in hypotheses_idx[-diff:]:
                unassigned_tracks.append(gt_centroid[list(gt_idx).index(idx)])
            print(unassigned_tracks)
            print(matched_clusters)

def nearest_search():
    pass

def filter_init(centroid, maxtrack_ID):
    # no match in gt 

    track_id = maxtrack_ID
    # state = pd.DataFrame[centroid[0], centroid[1], 0, 0, 0, 0]

    # std = np.arra
    # motion_cov = np.diag(std)
