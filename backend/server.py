from flask import Flask, request
from flask_cors import CORS
from parser import Parser
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from kalman import track_association
import matplotlib

matplotlib.use('Agg')
matplotlib.pyplot.axis('off')

APP = Flask(__name__)
CORS(APP)

@APP.route('/', methods=['POST'])
def input_frames():
	folder = request.get_json()['folder']
	try:
		frame_range = int(request.get_json()['start']), int(request.get_json()['end'])
	except ValueError:
		frame_range = None
	try:
		parser = Parser(folder, '{:06}.jpg', frame_range)
		output = candidate_small_objects_detection(parser)
		output = candidate_match_discrimination(output)
		track_association(output)
	except:
		return {
			'message': 'something went wrong'
		}, 400
	return {}

if __name__ == '__main__':
	APP.run(debug=True)