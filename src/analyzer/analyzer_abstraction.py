from abc import ABC, abstractmethod


class AnalyzerAbstraction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def analyze(self, stats_arr):
        pass
