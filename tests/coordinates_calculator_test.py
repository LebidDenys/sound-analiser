import utm

from src.coordinates_calculator.coordinates_calculator_implementation import CoordinatesCalculatorImplementation


def test_coordinates_calculator_success():
	coordinates_calculator = CoordinatesCalculatorImplementation()

	# Explosion point
	# "lat": 48.865,
	# "lng": 37.649
	res_x_y = utm.from_latlon(48.865, 37.649)
	print(res_x_y)
	test_items_xy = [{  # (47.51641000269241, 37.59999999992499)
		'x_coordinate': 394594.96382600203,  # distance 150000m
		'y_coordinate': 5263503.2599102575,
		'time': 440.79532779914793
	}, {  # (50.92000000345048, 40.36350000010464)
		'x_coordinate': 595838.8360418426,  # distance 300000m
		'y_coordinate': 5641813.950501725,
		'time': 881.590655598295
	}, {  # (44.86500000128786, 38.54899999998993)
		'x_coordinate': 464370.92708711646,  # distance 450000m
		'y_coordinate': 4968052.711467959,
		'time': 1322.3859833974438
	}]

	coordinates = coordinates_calculator.get_coordinates(test_items_xy)
	latlon = utm.to_latlon(coordinates[0], coordinates[1], 37, zone_letter="U")
	# Expected result: 400918.88624090137, 5413328.520821085
	assert type(coordinates) == list
	print(latlon)


test_coordinates_calculator_success()
