from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import init_tracks, predict, updateKalman, track_association

parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
frameBinary = candidate_small_objects_detection(parser)
frameCandiateClusters = candidate_match_discrimination(frameBinary, (200, 400), (0.85, 0.9), (20, 50), (0.45, 0.75))
frameTracks = init_tracks(frameCandiateClusters)
predictions = predict(frameTracks)
output = track_association(predictions, frameCandiateClusters)
# update = updateKalman(predictions)
# track_association(frames, )