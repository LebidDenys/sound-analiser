from abc import ABC, abstractmethod


class StreamInputterAbstraction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_stream(self):
        pass
