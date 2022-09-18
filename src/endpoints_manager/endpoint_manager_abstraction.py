from abc import ABC, abstractmethod


class EndpointsManagerAbstraction(ABC):
    def __init__(self,  *args, **kwargs):
        pass

    @abstractmethod
    def start_streams(self):
        pass

    @abstractmethod
    def _stream_callback(self, stats_arr):
        pass

    @abstractmethod
    def _handle_explosion(self, endpoint_index, explosion_endpoint, time_ms):
        pass
