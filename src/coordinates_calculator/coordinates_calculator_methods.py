import math

from numpy import float64


def calculate_delta(vector_one_coordinates, vector_two_coordinates):
	vector_one_x, vector_one_y = vector_one_coordinates
	vector_two_x, vector_two_y = vector_two_coordinates
	return vector_one_x * vector_two_y - vector_one_y * vector_two_x


def calculate_base_len(coordinates_a, coordinates_b):
	x_delta_square = math.pow(coordinates_a["x_coordinate"] - coordinates_b["x_coordinate"], 2)
	y_delta_square = math.pow(coordinates_a["y_coordinate"] - coordinates_b["y_coordinate"], 2)

	return math.pow(x_delta_square + y_delta_square, .5)


def calculate_time_difference(timeA, timeB):
	return (timeA - timeB) / 1000


def calculate_delta_for_axis(vector_one_coordinates, vector_two_coordinates, base_one_center_coordinates,
							 base_two_center_coordinates, axisX=True):
	vector_one_x, vector_one_y = vector_one_coordinates
	vector_two_x, vector_two_y = vector_two_coordinates
	base_one_center_x, base_one_center_y = base_one_center_coordinates
	base_two_center_x, base_two_center_y = base_two_center_coordinates

	vector_axis_x, vector_axis_y = (vector_one_x, vector_two_x) if axisX else (vector_one_y, vector_two_y)
	return vector_axis_x * (base_two_center_x * vector_two_y - base_two_center_y * vector_one_x) \
		   - vector_axis_y * (base_one_center_x * vector_one_y - base_one_center_y * vector_one_x)


def calculate_base_center_coordinates(coordinates_a, coordinates_b):
	return [
		(coordinates_a["x_coordinate"] + coordinates_b["x_coordinate"]) / 2,
		(coordinates_a["y_coordinate"] + coordinates_b["y_coordinate"]) / 2
	]


def calculate_normal_coordinates(coordinates_a, coordinates_b, base_len):
	return [
		(coordinates_b["y_coordinate"] - coordinates_a["y_coordinate"]) / base_len,
		(coordinates_b["x_coordinate"] - coordinates_a["x_coordinate"]) / base_len
	]


def calculate_vector_h(normal_coordinates, betta):
	normal_x, normal_y = normal_coordinates
	cos_betta = math.cos(betta)
	sin_betta = math.sin(betta)
	return [
		normal_x * cos_betta - normal_y * sin_betta,
		normal_y * cos_betta + normal_x * sin_betta,
	]


def get_sound_speed(temperate_celsius=15):
	KELVINS_SHIFT = 273.15
	TEMPERATURE_RATIO = 1.4
	GAS_CONSTANT = 287.05287

	SOUND_SPEED_FOR_DEFAULT_TEMPERATURE_METERS = 340.293988026
	return SOUND_SPEED_FOR_DEFAULT_TEMPERATURE_METERS if temperate_celsius == 15 \
		else math.sqrt(TEMPERATURE_RATIO * GAS_CONSTANT * (temperate_celsius + KELVINS_SHIFT))


def validate(variables_values, variable_name):
	for variable in variables_values:
		if variable is None or (type(variable) != float and type(variable) != float64):
			raise Exception(f"Invalid value: {variable}, for variable: {variable_name}")


def calculate_betta(time_diff, base_len):
	alpha = get_sound_speed()
	angle_sinus = (time_diff * alpha) / base_len

	if type(angle_sinus) != float or angle_sinus > 1 or angle_sinus < -1:
		raise Exception(
			f"Invalid calculation params: sinus = {angle_sinus}, time difference = {time_diff}, base length = {base_len}")

	return math.asin(angle_sinus)
