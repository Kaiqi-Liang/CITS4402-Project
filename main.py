from parser import Parser
from detection import candidate_small_objects_detection
parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
candidate_small_objects_detection(2, parser)