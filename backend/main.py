from parser import Parser
from detection import candidate_small_objects_detection
from match import region_growing, thresholds_calibration, candidate_match_discrimination
from kalman import init_tracks, predict_tracks, updated_tracks, track_association, kalman

# Parser to read files in
parser = Parser('mot/car/001', '{:06}.jpg', (1, 10))

# Small object detection to output binary image of candidate moving objects
frameBinary = candidate_small_objects_detection(parser)

# Region growing to grow candidate objects
frameBinary = region_growing(frameBinary)

# thresholds_calibration(parser, frameBinary)

# Candidate match discrimination to apply morphological cues to binary image
hypothesis = candidate_match_discrimination(frameBinary, (0, 100), (0.45, 0.9), (10, 50), (0.45, 0.75))

# Kalman to track candidate small objects across frames 
output = kalman(hypothesis)

# initializedtracks = init_tracks(hypothesis)
# predictions = predict_tracks(initializedtracks)
# matched_pairs = track_association(predictions, hypothesis[1])
# update = updated_tracks(matched_pairs)
