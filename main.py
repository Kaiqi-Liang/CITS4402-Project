from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
import matplotlib.pyplot as plt

parser = Parser('mot/car/001', '{:06}.jpg', (1, 4))
images = candidate_small_objects_detection(parser)
region_grow = candidate_match_discrimination(images)

fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(region_grow[0][1], cmap='gray')
plt.title("REGION GROW")


fig.add_subplot(rows, columns, 2)
plt.imshow(region_grow[0][0], cmap='gray')
plt.title("ORIGINAL BINARY IMAGE")

plt.show()