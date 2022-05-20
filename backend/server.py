from flask import Flask, request
from flask_cors import CORS
from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination, region_growing, thresholds_calibration
from kalman import track_association
import matplotlib

matplotlib.use('Agg')
matplotlib.pyplot.axis('off')

APP = Flask(__name__)
CORS(APP)

@APP.route('/calibration', methods=['POST'])
def calibration():
	data = request.get_json()
	folder = data['folder']
	frames = int(data['frames'])
	try:
		frame_range = int(data['start']), int(data['end'])
	except ValueError:
		frame_range = None
	try:
		parser = Parser(folder, '{:06}.jpg', frame_range)
		frameBinary = candidate_small_objects_detection(parser, frames)
		frameBinary = region_growing(frameBinary)
		thresholds_calibration(parser, frameBinary)
	except:
		return {
			'message': 'something went wrong'
		}, 400
	return {}

@APP.route('/track', methods=['POST'])
def start_tracking():
	data = request.get_json()
	folder = data['folder']
	frames = int(data['frames'])
	areaTh = float(data['areaUpperTh']), float(data['areaLowerTh'])
	extentTh = float(data['extentUpperTh']), float(data['extentLowerTh'])
	majorAxisTh = float(data['majorAxisUpperTh']), float(data['majorAxisLowerTh'])
	eccentricityTh = float(data['eccentricityUpperTh']), float(data['eccentricityLowerTh'])
	try:
		frame_range = int(data['start']), int(data['end'])
	except ValueError:
		frame_range = None
	try:
		parser = Parser(folder, '{:06}.jpg', frame_range)
		frameBinary = candidate_small_objects_detection(parser, frames)
		frameBinary = region_growing(frameBinary)
		output = candidate_match_discrimination(frameBinary, areaTh, extentTh, majorAxisTh, eccentricityTh)
	except:
		return {
			'message': 'something went wrong'
		}, 400
	return {}

if __name__ == '__main__':
	APP.run(debug=True)