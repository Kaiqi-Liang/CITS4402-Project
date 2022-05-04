from parse import Parser
parser = Parser('mot/car/001', '{:06}.jpg', (1, 3))
print(parser.get_frame_range())
print(parser.load_frame(1))