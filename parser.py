from cv2 import imread
class Parser:
	def __init__(self, folder: str, template: str, frame_range: tuple[int, int]) -> None:
		self.start, self.end = frame_range
		self.frames = [f'{folder}/img/{template.format(frame)}' for frame in range(self.start, self.end)]

	def get_frame_range(self) -> tuple[int, int]:
		return self.start, self.end

	def load_frame(self, index: int):
		return imread(self.frames[index - 1])