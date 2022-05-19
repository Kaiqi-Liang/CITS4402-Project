from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import init_tracks, predict, updateKalman, track_association

parser = Parser('mot/car/001', '{:06}.jpg', (1, 50))
frameBinary = candidate_small_objects_detection(parser)
frameCandiateClusters = candidate_match_discrimination(frameBinary, (0, 100), (0.45, 0.9), (10, 50), (0.45, 0.75))
initializedtracks = init_tracks(frameCandiateClusters)
predictions = predict(initializedtracks)
matched_pairs = track_association(predictions, frameCandiateClusters)
update = updateKalman(matched_pairs)
# track_association(frames, )