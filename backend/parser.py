import numpy as np
import pandas as pd
import os
import cv2
class Parser:
	def __init__(self, folder: str, template: str, frame_range: tuple[int, int] = None) -> None:
		self.folder = f'{folder}/img/'
		self.frames = sorted(os.listdir(self.folder))
		self.start = int(self.frames[0][:self.frames[0].find('.')])
		self.end = int(self.frames[-1][:self.frames[-1].find('.')])
		if frame_range:
			start, end = frame_range
			if start < self.start or end > self.end or end - start < 3:
				raise ValueError
			self.start, self.end = start, end
			self.frames = [f'{template.format(frame)}' for frame in range(self.start, self.end)]

		self.gt = []
		self.gt_centroid = []
		self.read_gt(f'{folder}/gt/gt.txt', self.gt)
		self.read_gt(f'{folder}/gt/gt_centroid.txt', self.gt_centroid)

	def read_gt(self, csv: str, gt: list):
		df = pd.read_csv(csv, usecols=range(6), names=['Frame No', 'Track ID', 'X', 'Y', 'Width', 'Height'])
		for i in range(self.start, self.end):
			gt.append(df[df['Frame No'] == i].iloc[:, 1:].values.tolist())

	def get_frame_range(self) -> tuple[int, int]:
		return self.start, self.end

	def load_frame(self, frame: int) -> np.ndarray:
		return cv2.imread(f'{self.folder}{self.frames[frame - self.start]}')

	def get_gt(self, frame: int) -> list[list[int]]:
		return self.gt[frame - self.start]

	def get_gt_centroid(self, frame: int) -> list[list[int]]:
		return self.gt_centroid[frame - self.start]