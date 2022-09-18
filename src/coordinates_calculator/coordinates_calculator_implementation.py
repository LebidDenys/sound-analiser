from .coordinates_calculator_abstraction import CoordinatesCalculatorAbstraction
from .coordinates_calculator_methods import validate, calculate_base_len, calculate_normal_coordinates, \
	calculate_base_center_coordinates, calculate_time_difference, calculate_betta, calculate_vector_h, \
	calculate_delta, calculate_delta_for_axis


class CoordinatesCalculatorImplementation(CoordinatesCalculatorAbstraction):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def get_coordinates(self, points_arr):
		"""
		Get coordinates for array of points
		Args:
			points_arr: array of points with coordinates and time

		Returns: array with two items, first one is X coordinate, second one is Y
		"""
		result_data = {
			"x_sum": 0, "y_sum": 0, "items": 0
		}

		for i in range(len(points_arr) - 2):
			result_data["x_sum"], result_data["y_sum"] = self.calculate(points_arr[i], points_arr[i + 1], points_arr[i + 2])
			result_data["items"] += 1

		x_res = result_data["x_sum"] / result_data["items"]
		y_res = result_data["y_sum"] / result_data["items"]

		return [x_res, y_res]

	def calculate(self, sound_meter_a, sound_meter_b, sound_meter_c):
		base_len_1 = calculate_base_len(sound_meter_a, sound_meter_b)
		base_len_2 = calculate_base_len(sound_meter_b, sound_meter_c)
		base_len_3 = calculate_base_len(sound_meter_a, sound_meter_c)
		validate([base_len_1, base_len_2, base_len_3], 'Base Lengths')

		base_center_coordinates_1 = calculate_base_center_coordinates(sound_meter_a, sound_meter_b)
		base_center_coordinates_2 = calculate_base_center_coordinates(sound_meter_b, sound_meter_c)
		base_center_coordinates_3 = calculate_base_center_coordinates(sound_meter_a, sound_meter_c)
		validate(base_center_coordinates_1 + base_center_coordinates_2 + base_center_coordinates_3,
				 'Base Center Coordinates')

		normal_coordinates_1 = calculate_normal_coordinates(sound_meter_a, sound_meter_b, base_len_1)
		normal_coordinates_2 = calculate_normal_coordinates(sound_meter_b, sound_meter_c, base_len_2)
		normal_coordinates_3 = calculate_normal_coordinates(sound_meter_a, sound_meter_c, base_len_3)
		validate(normal_coordinates_1 + normal_coordinates_2 + normal_coordinates_3, 'Normal Vector Coordinates')

		time_diff_1 = calculate_time_difference(sound_meter_a["time"], sound_meter_b["time"])
		time_diff_2 = calculate_time_difference(sound_meter_b["time"], sound_meter_c["time"])
		time_diff_3 = calculate_time_difference(sound_meter_a["time"], sound_meter_c["time"])
		validate([time_diff_1, time_diff_2, time_diff_3], 'Time Difference')

		betta_angle_1 = calculate_betta(time_diff_1, base_len_1)
		betta_angle_2 = calculate_betta(time_diff_2, base_len_2)
		betta_angle_3 = calculate_betta(time_diff_3, base_len_3)
		validate([betta_angle_1, betta_angle_2, betta_angle_3], 'Betta angle')

		vector_coordinates_1 = calculate_vector_h(normal_coordinates_1, betta_angle_1)
		vector_coordinates_2 = calculate_vector_h(normal_coordinates_2, betta_angle_2)
		vector_coordinates_3 = calculate_vector_h(normal_coordinates_3, betta_angle_3)
		validate(vector_coordinates_1 + vector_coordinates_2 + vector_coordinates_3, 'Vector H coordinates')

		delta_1 = calculate_delta(vector_coordinates_1, vector_coordinates_2)
		delta_2 = calculate_delta(vector_coordinates_1, vector_coordinates_3)
		delta_3 = calculate_delta(vector_coordinates_2, vector_coordinates_3)
		validate([delta_1, delta_2, delta_3], 'Delta')

		delta_x_1 = calculate_delta_for_axis(vector_coordinates_1, vector_coordinates_2, base_center_coordinates_1, base_center_coordinates_2, True)
		delta_x_2 = calculate_delta_for_axis(vector_coordinates_1, vector_coordinates_3, base_center_coordinates_1, base_center_coordinates_3, True)
		delta_x_3 = calculate_delta_for_axis(vector_coordinates_2, vector_coordinates_3, base_center_coordinates_2, base_center_coordinates_3, True)
		validate([delta_x_1, delta_x_2, delta_x_3], 'Delta for X Axis')

		delta_y_1 = calculate_delta_for_axis(vector_coordinates_1, vector_coordinates_2, base_center_coordinates_1, base_center_coordinates_2, False)
		delta_y_2 = calculate_delta_for_axis(vector_coordinates_1, vector_coordinates_3, base_center_coordinates_1, base_center_coordinates_3, False)
		delta_y_3 = calculate_delta_for_axis(vector_coordinates_2, vector_coordinates_3, base_center_coordinates_2, base_center_coordinates_3, False)
		validate([delta_y_1, delta_y_2, delta_y_3], 'Delta for Y Axis')

		x_axis_h_1 = delta_x_1 / delta_1
		x_axis_h_2 = delta_x_2 / delta_2
		x_axis_h_3 = delta_x_3 / delta_3
		validate([x_axis_h_1, x_axis_h_2, x_axis_h_3], 'X coordinates')

		y_axis_h_1 = delta_y_1 / delta_1
		y_axis_h_2 = delta_y_2 / delta_2
		y_axis_h_3 = delta_y_3 / delta_3
		validate([y_axis_h_1, y_axis_h_2, y_axis_h_3], 'Y coordinates')

		x_result = (x_axis_h_1 + x_axis_h_2 + x_axis_h_3) / 3
		y_result = (y_axis_h_1 + y_axis_h_2 + y_axis_h_3) / 3
		validate([x_result, y_result], 'Result Coordinates')

		return [x_result, y_result]
