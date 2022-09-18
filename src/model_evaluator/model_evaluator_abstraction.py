from abc import ABC, abstractmethod


class ModelEvalAbstraction(ABC):

    @abstractmethod
    def stats_callback(self, stats):
        pass

    @abstractmethod
    def eval_model(self):
        pass

