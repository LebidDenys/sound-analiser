from abc import ABC, abstractmethod


class ModelTrainerAbstraction(ABC):

    @abstractmethod
    def stats_callback(self, stats):
        pass

    @abstractmethod
    def train_model(self):
        pass

