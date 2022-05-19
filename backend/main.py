from parser import Parser
from detection import candidate_small_objects_detection
from match import region_growing, thresholds_calibration, candidate_match_discrimination
from kalman import init_tracks, predict, updateKalman, track_association

parser = Parser('mot/car/001', '{:06}.jpg', (1, 87))
frameBinary = candidate_small_objects_detection(parser)
frameBinary = region_growing(frameBinary)
# thresholds_calibration(parser, frameBinary)
frameCandiateClusters = candidate_match_discrimination(frameBinary, (0, 50), (0.25, 1), (2.5, 12.5), (0.4, 1))
initializedtracks = init_tracks(frameCandiateClusters)
predictions = predict(initializedtracks)
matched_pairs = track_association(predictions, frameCandiateClusters)
update = updateKalman(matched_pairs)
# track_association(frames, )