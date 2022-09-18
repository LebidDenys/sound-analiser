import logging
import time

from threading import Thread

import utm

from .endpoint_manager_abstraction import EndpointsManagerAbstraction
from src.analyzer.analyzer_abstraction import AnalyzerAbstraction
from src.coordinates_calculator.coordinates_calculator_abstraction import CoordinatesCalculatorAbstraction
from src.stream_inputter.stream_inputter_implementation import StreamInputterImplementation


def _map_config_to_utm(index, config_endpoint):
    (x, y, lng_index, lat_letter) = utm.from_latlon(config_endpoint['lat'], config_endpoint['lng'])
    return {
        'id': index,
        'url': config_endpoint['url'],
        'x_coordinate': x,
        'y_coordinate': y
    }


class EndpointsManagerImplementation(EndpointsManagerAbstraction):
    def __init__(self, stat_calc, config, analyzer: AnalyzerAbstraction,
                 coordinates_calculator: CoordinatesCalculatorAbstraction, coordinates_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_calc = stat_calc
        self.config = config
        self.analyzer = analyzer
        self.coordinates_calculator = coordinates_calculator
        self.coordinates_callback = coordinates_callback
        self.explosions_array = []

        self.endpoints = [_map_config_to_utm(index, config_endpoint)
                          for index, config_endpoint in enumerate(self.config['endpoints'])]

    def start_streams(self):
        """
        Start separated stream for each of endpoints from config
        """
        logging.info('Starting streams')

        threads = []
        for endpoint_index, endpoint in enumerate(self.endpoints):
            streamer = StreamInputterImplementation(
                stream_url=endpoint['url'],
                stat_calc=self.stat_calc,
                packet_size=self.config['packet_size'],
                sample_rate=self.config['sample_rate'],
                callback_method=self._stream_callback,
            )
            threads.append(Thread(target=streamer.read_stream, args=[endpoint_index]))
            threads[-1].start()
        for thread in threads:
            thread.join()

    def _stream_callback(self, stats_arr, *args, **kwargs):
        """
        Would receive stats_arr for each of stream chunk
        """
        explosion_time_ms = time.time_ns() / 1000 # should be before analyzation, for time accuracy
        endpoint_index = args[0]
        prediction = self.analyzer.analyze(stats_arr)[0]
        if prediction > self.config['prediction_threshold']:
            explosion_endpoint = self.endpoints[endpoint_index]
            self._handle_explosion(endpoint_index, explosion_endpoint, explosion_time_ms)

    def _handle_explosion(self, endpoint_index, explosion_endpoint, time_ms):
        filtered_explosions = [{
            'id': endpoint_index,
            'x_coordinate': explosion_endpoint['x_coordinate'],
            'y_coordinate': explosion_endpoint['y_coordinate'],
            'time': time_ms
        }]
        for explosion in self.explosions_array:
            is_explosion_outdated \
                = explosion['time'] + self.config['explosion_cache_lifetime_ms'] < (time.time_ns() / 1000)
            is_same_endpoint = explosion['id'] == endpoint_index
            if not is_explosion_outdated and not is_same_endpoint:
                filtered_explosions.append(explosion)
        self.explosions_array = filtered_explosions
        if len(self.explosions_array) > 2:
            easting, northing = self.coordinates_calculator.get_coordinates(self.explosions_array)
            zoneinfo = self.config['zoneinfo']
            lat, lng = utm.to_latlon(easting, northing,
                                                 zoneinfo['longitudinal_index'], zoneinfo['latitudinal_letter'])
            self.coordinates_callback(lat, lng)


