from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import scikit-image as skimage
def candidate_match_discrimination(parser: Parser, candidate_small_objects):
	for object in candidate_small_objects:
		label_image = skimage.measure.label(object)
		skimage.measure.regionprops(label_imag)
