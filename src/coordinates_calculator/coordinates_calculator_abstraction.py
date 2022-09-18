from abc import ABC, abstractmethod


class CoordinatesCalculatorAbstraction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    """
    Returns: array with two items, first one is X coordinate, second one is Y
    """
    @abstractmethod
    def get_coordinates(self, points_arr):
        pass