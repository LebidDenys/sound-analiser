import ffmpeg
import numpy
import logging

from .stream_inputter_abstraction import StreamInputterAbstraction


class StreamInputterImplementation(StreamInputterAbstraction):

    def __init__(self, stream_url, stat_calc, packet_size, sample_rate,
                 callback_method, close_callback_method=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_url = stream_url
        self.stat_calc = stat_calc
        self.packet_size = packet_size  # 4096 * 64
        self.sample_rate = sample_rate
        self.callback_method = callback_method
        self.close_callback_method = close_callback_method

    def read_stream(self, *args, **kwargs):
        """
        Read stream from url (could be file path) using ffmpeg,
        stream would be read by chunks size of PACKET_SIZE,
        then transformed to numpy array and analyzed,
        stats would be passed to callback method as argument
        """
        logging.info(f'Starting stream from url {self.stream_url}')
        process = (
            ffmpeg
            .input(self.stream_url)
            .output('pipe:', format='s16le', sample_rate=f'{self.sample_rate}', allowed_media_types='audio')
            .run_async(pipe_stdout=True)
        )
        index = 0
        while process.poll() is None:
            packet = process.stdout.read(self.packet_size)
            audio_array = numpy.frombuffer(packet, dtype="int16")

            stats = self.stat_calc.calculate_stats(audio_array, index)
            index += 1

            self.callback_method(stats, *args, **kwargs)

        # to let know that stream is finished
        if self.close_callback_method is not None:
            self.close_callback_method()
