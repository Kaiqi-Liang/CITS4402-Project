from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination

parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
objects = candidate_small_objects_detection(parser)
candidate_match_discrimination(parser, objects)