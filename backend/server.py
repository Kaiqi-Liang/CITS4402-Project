from flask import Flask, request
from flask_cors import CORS, cross_origin
from detection import candidate_small_objects_detection
from match import candidate_match_discrimination
from parser import Parser

APP = Flask(__name__)
CORS(APP)

@APP.route('/', methods=['POST'])
def input_frames():
	folder = request.get_json()['folder']
	start = int(request.get_json()['start'])
	end = int(request.get_json()['end'])
	try:
		parser = Parser(folder, '{:06}.jpg', (start, end))
		images = candidate_small_objects_detection(parser)
		candidate_match_discrimination(parser, images)
	except:
		return {
			'message': 'something went wrong'
		}, 400
	return {}

if __name__ == '__main__':
	APP.run(debug=True)