from kalman import track_association
from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination

parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
output = candidate_small_objects_detection(parser)
output = candidate_match_discrimination(output)
track_association(output)