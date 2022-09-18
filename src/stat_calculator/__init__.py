import warnings

import librosa
import numpy as np
from scipy.stats import skew, kurtosis, median_abs_deviation
from sorcery import dict_of


class StatCalc:
    def __init__(self, timing_param, sample_rate, add_timing=True):
        """

        Args:
            timing_param (float): length of the chunk in seconds
            sample_rate (int): audio sample rate in Hz

        Returns:

        """
        self.timing_param = timing_param
        self.sample_rate = sample_rate
        self.add_timing = add_timing

    def calculate_stats(self, audio_array,
                        index):  # TODO: limit number of features generated only to the columns that were read
        """
        Calculate statistics for given array.
        """
        # audio_array = librosa.util.normalize(audio_array.astype('float'))

        # calculate strongest frequencies
        w = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(w))

        freqs = freqs[w.argsort()]
        w = w[w.argsort()]

        std = audio_array.std()
        var = audio_array.var()
        avg = sum(audio_array, 0.0) / len(audio_array)
        mean = audio_array.mean()
        maxVal = max(audio_array)
        diff = maxVal - mean
        skewness = skew(audio_array)
        kurtosis_ = kurtosis(audio_array)
        max_deviation = max(abs(el - avg) for el in audio_array)
        median_abs_deviation_ = median_abs_deviation(audio_array)
        abs_mean = np.abs(audio_array).mean()
        rms = np.sqrt(np.sum(audio_array ** 2) / audio_array.shape[0])
        p_t_p = maxVal - min(audio_array)
        crest_f = 0 if rms == 0 else maxVal / rms
        energy = np.sum(audio_array ** 2) / audio_array.shape[0]
        fft_1 = freqs[-1]
        fft_2 = freqs[-2]
        fft_3 = freqs[-3]
        fft_4 = freqs[-5]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # calculate centroids and their basic stats so that we have aggs
            centroids = librosa.feature.spectral_centroid(y=audio_array.astype('float'), sr=self.sample_rate)
            # calculate spectral bandwidth and their basic stats so that we have aggs
            sp_bandwidth = librosa.feature.spectral_bandwidth(y=audio_array.astype('float'), sr=self.sample_rate)

        centroids_ptp = centroids.max() - centroids.min()
        centroids_mean = centroids.mean()
        centroids_std = np.std(centroids)
        centroids_kurt = kurtosis(centroids, axis=None)
        centroids_skew = skew(centroids, axis=None)

        sp_bandwidth_ptp = sp_bandwidth.max() - sp_bandwidth.min()
        sp_bandwidth_mean = sp_bandwidth.mean()
        sp_bandwidth_std = np.std(sp_bandwidth)
        sp_bandwidth_kurt = kurtosis(sp_bandwidth, axis=None)
        sp_bandwidth_skew = skew(sp_bandwidth, axis=None)

        stats = dict_of(
            mean,
            maxVal,
            diff,
            std,
            var,
            skewness,
            kurtosis_,
            median_abs_deviation_,
            max_deviation,
            centroids_ptp,
            centroids_mean,
            centroids_std,
            centroids_kurt,
            centroids_skew,
            sp_bandwidth_ptp,
            sp_bandwidth_mean,
            sp_bandwidth_std,
            sp_bandwidth_kurt,
            sp_bandwidth_skew,
            fft_1,
            fft_2,
            fft_3,
            fft_4,
            abs_mean,
            rms,
            p_t_p,
            crest_f,
            energy
        )

        if self.add_timing:
            l_timing = index * self.timing_param
            h_timing = (index + 1) * self.timing_param
            stats['approximate_timing'] \
                = f'{int(l_timing / 60)}:{round(l_timing % 60, 1)} - {int(h_timing / 60)}:{round(h_timing % 60, 1)}'

            stats['start_time'] = l_timing
            stats['end_time'] = h_timing

        return stats
