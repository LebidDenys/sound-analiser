import argparse
import logging

from src.io_manager import load_pickle, load_list, read_yaml
from src.analyzer.analyzer_implementation import AnalyzerImplementation
from src.coordinates_calculator.coordinates_calculator_implementation import CoordinatesCalculatorImplementation
from src.endpoints_manager.endpoint_manager_implementation import EndpointsManagerImplementation
from src.model_evaluator.model_evaluator_implementation import ModelEvalImplementation
from src.model_trainer.model_trainer_implementation import ModelTrainerImplementation
from src.stat_calculator import StatCalc
from src.stream_inputter.stream_inputter_implementation import StreamInputterImplementation

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
	parser = argparse.ArgumentParser("Welcome to sound analyzer")
	parser.add_argument("--mode", type=str, help="evaluate/predict/train", default='predict')

	parser.add_argument("--url", type=str, help="Required only for train mode, pass None for other modes")

	return parser.parse_args()


def coordinates_callback(lat, lng):
	# TODO: add explosion handling here
	message = '\n' + f'Explosion coordinates: latitude - {lat}, longitude - {lng}'.center(40, '#') + '\n'
	logging.info(message)
	print(message)


if __name__ == "__main__":
	logging.basicConfig(filename='./logging/logs.txt',
						level=logging.INFO,
						format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

	args = parse_args()
	config = read_yaml()

	model_id = config['model_id']
	packet_size = config['packet_size']
	sample_rate = config['sample_rate']

	MODEL_OUTPUT_PATH = f'./output/results{model_id}.csv'
	COLS_PATH = f'./artifacts/columns/xgb{model_id}.txt'
	MODEL_PATH = f'./artifacts/models/xgb{model_id}.pkl'

	mode = "predict"
	# mode = args.mode

	logging.info(f'Application starting in {mode} mode.\n Config values, model output path - {MODEL_OUTPUT_PATH},\n'
				 f'cols path - {COLS_PATH},\n model path - {MODEL_PATH}')

	stat_calc = StatCalc(
		timing_param=packet_size / (4 * sample_rate),
		sample_rate=sample_rate
	)

	if mode == "train":
		trainer = ModelTrainerImplementation(
			labels_path=config['labels_path'],
			model_path=MODEL_PATH,
			cols_path=COLS_PATH,
			train_result_path=MODEL_OUTPUT_PATH,
			model_id=model_id
		)
		streamer = StreamInputterImplementation(
			stream_url=args.url,
			stat_calc=stat_calc,
			packet_size=packet_size,
			sample_rate=sample_rate,
			callback_method=trainer.stats_callback,
			close_callback_method=trainer.train_model
		)
		streamer.read_stream()

	elif mode == "evaluate":
		evaluator = ModelEvalImplementation(
			labels_path=config['labels_path'],
			model_path=MODEL_PATH,
			cols_path=COLS_PATH,
			eval_result_path=MODEL_OUTPUT_PATH
		)
		streamer = StreamInputterImplementation(
			stream_url=args.url,
			stat_calc=stat_calc,
			packet_size=packet_size,
			sample_rate=sample_rate,
			callback_method=evaluator.stats_callback,
			close_callback_method=evaluator.eval_model
		)
		streamer.read_stream()

	elif mode == "predict":
		model = load_pickle(MODEL_PATH)
		cols = load_list(COLS_PATH)
		analyzer = AnalyzerImplementation(
			model=model,
			cols=cols
		)
		coordinates_calculator = CoordinatesCalculatorImplementation()
		endpoints_manager = EndpointsManagerImplementation(
			stat_calc=stat_calc,
			config=config,
			analyzer=analyzer,
			coordinates_calculator=coordinates_calculator,
			coordinates_callback=coordinates_callback
		)
		endpoints_manager.start_streams()
	else:
		raise RuntimeError(f"Unrecognized mode {mode}")
