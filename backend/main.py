from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import init_tracks, predict

parser = Parser('mot/car/001', '{:06}.jpg', (1, 6))
frameBinary = candidate_small_objects_detection(parser)
frameCandiateClusters = candidate_match_discrimination(frameBinary)
frameTracks = init_tracks(frameCandiateClusters)
predictions = predict(frameTracks)
# track_association(frames, )