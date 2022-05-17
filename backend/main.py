from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import init_tracks, predict

parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
output = candidate_small_objects_detection(parser)
output = candidate_match_discrimination(output)
output = init_tracks(output)
print(output)
# predictions = predict(tracks)
# track_association(frames, )