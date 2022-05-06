import numpy as np
import os
import cv2
class Parser:
	def __init__(self, folder: str, template: str, frame_range: tuple[int, int] = None) -> None:
		self.folder = f'{folder}/img/'
		if frame_range:
			self.start, self.end = frame_range
			self.frames = [f'{self.folder}{template.format(frame)}' for frame in range(self.start, self.end)]
		else:
			self.frames = sorted(os.listdir(self.folder))
			self.start = int(self.frames[0][:self.frames[0].find('.')])
			self.end = int(self.frames[-1][:self.frames[-1].find('.')])

	def get_frame_range(self) -> tuple[int, int]:
		return self.start, self.end

	def load_frame(self, index: int) -> np.ndarray:
		return cv2.imread(self.frames[index - 1])