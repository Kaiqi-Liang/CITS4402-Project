import os
import pandas
import cv2

def parse(folder: str, start: int, end: int):
	os.chdir(folder)