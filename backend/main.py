from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import kalman

# Parser to read files in
parser = Parser('mot/car/001', '{:06}.jpg', (1, 10))

# Small object detection to output binary image of candidate moving objects
frameBinary = candidate_small_objects_detection(parser)

# Candidate match discrimination to apply morphological cues to binary image
hypothesis = candidate_match_discrimination(frameBinary, (0, 100), (0.45, 0.9), (10, 50), (0.45, 0.75))

# Kalman to track candidate small objects across frames 
kalman(hypothesis)
