from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.fft import fft
import numpy as np


class Shifted1D_LBP:
    def __init__(self):
        self.has_parkinson = None
        self.no_neightbours = None
        self.no_central = None
        self.time = None
        self.signal_to_preprocess = None

        self.preprocessed_signal = None
        self.entropy = None
        self.energy = None
        self.correlation = None
        self.cov = None
        self.kurtosis = None
        self.skewness = None

        self.preprocessed_frequency_signal = None
        self.freq_entropy = None
        self.freq_energy = None
        self.freq_correlation = None
        self.freq_cov = None
        self.freq_kurtosis = None
        self.freq_skewness = None

    def shifting(self):
        self.preprocessed_signal = np.array(self.shift_signal(self.signal_to_preprocess))

    def shift_signal(self, signal):
        preprocessed_signal = []
        for i in range(self.no_central, len(signal) - self.no_neightbours + self.no_central - 1):
            central_point = []
            for idx in range(i - self.no_central, i + self.no_neightbours - self.no_central + 1):
                if idx == i:
                    continue
                t = signal[idx] - signal[i]
                central_point.append('1' if t >= 0 else '0')
            binary_number = ''.join(central_point)
            preprocessed_signal.append(self.convert_bin_to_int(binary_number))
        return preprocessed_signal

    def shifting_freq(self):
        frequency_signal_to_preprocess = fft(self.signal_to_preprocess)
        self.preprocessed_frequency_signal = np.array(self.shift_signal(frequency_signal_to_preprocess))

    @staticmethod
    def convert_bin_to_int(str_number):
        return int(str_number, 2)

    def calculate_entropy(self, signal):
        sum_squared_sqrt = np.sqrt(np.sum(np.square(signal)))
        entropy_function = lambda value: (value / sum_squared_sqrt) * np.log(value / sum_squared_sqrt)
        entropy = np.array([entropy_function(x) for x in signal])
        self.entropy = np.nansum(entropy)
        return self.entropy

    def calculate_entropy_time(self):
        self.entropy = self.calculate_entropy(self.preprocessed_signal)
        return self.entropy

    def calculate_entropy_freq(self):
        self.freq_entropy = self.calculate_entropy(self.preprocessed_frequency_signal)
        return self.freq_entropy

    def calculate_energy(self, signal):
        sum_squared = np.square(signal)
        energy = np.array([x * self.time[i + self.no_central] for i, x in enumerate(sum_squared)])
        return np.nansum(energy)

    def calculate_energy_time(self):
        self.energy = self.calculate_energy(self.preprocessed_signal)
        return self.energy

    def calculate_energy_freq(self):
        self.freq_energy = self.calculate_energy(self.preprocessed_frequency_signal)
        return self.freq_energy

    def calculate_correlation(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        correlation = np.array(
            [(i * x - signal_mean) / signal_std for i, x in enumerate(signal)])
        return np.nansum(correlation)

    def calculate_correlation_time(self):
        self.correlation = self.calculate_correlation(self.preprocessed_signal)
        return self.correlation

    def calculate_correlation_freq(self):
        self.freq_correlation = self.calculate_correlation(self.preprocessed_frequency_signal)
        return self.freq_correlation

    def calculate_coefficient_of_variation(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        return signal_std / signal_mean

    def calculate_coefficient_of_variation_time(self):
        self.cov = self.calculate_coefficient_of_variation(self.preprocessed_signal)
        return self.cov

    def calculate_coefficient_of_variation_freq(self):
        self.freq_cov = self.calculate_coefficient_of_variation(self.preprocessed_frequency_signal)
        return self.freq_cov

    def calculate_kurtosis(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        kurtosis_part = np.array([(i * x - signal_mean) ** 4 for i, x in enumerate(signal)])
        return np.nansum(kurtosis_part) / signal.size / (signal_std ** 4)

    def calculate_kurtosis_time(self):
        self.kurtosis = self.calculate_kurtosis(self.preprocessed_signal)
        return self.kurtosis

    def calculate_kurtosis_freq(self):
        self.freq_kurtosis = self.calculate_kurtosis(self.preprocessed_frequency_signal)
        return self.freq_kurtosis

    def calculate_skewness(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        skewness_part = np.array(
            [((i * x - signal_mean) / signal_std) ** 3 for i, x in enumerate(signal)])
        return signal.size / (
                (signal.size - 1) * (signal.size - 2)) * np.nansum(
            skewness_part)

    def calculate_skewness_time(self):
        self.skewness = self.calculate_skewness(self.preprocessed_signal)
        return self.skewness

    def calculate_skewness_freq(self):
        self.freq_skewness = self.calculate_skewness(self.preprocessed_frequency_signal)
        return self.freq_skewness

    def calculate_features(self):
        self.calculate_entropy_time()
        self.calculate_energy_time()
        self.calculate_correlation_time()
        self.calculate_coefficient_of_variation_time()
        self.calculate_kurtosis_time()
        self.calculate_skewness_time()

    def calculate_features_freq(self):
        self.calculate_entropy_freq()
        self.calculate_energy_freq()
        self.calculate_correlation_freq()
        self.calculate_coefficient_of_variation_freq()
        self.calculate_kurtosis_freq()
        self.calculate_skewness_freq()

    def get_entropy(self):
        return self.entropy

    def get_energy(self):
        return self.energy

    def get_correlation(self):
        return self.correlation

    def get_coefficient_of_variation(self):
        return self.cov

    def get_kurtosis(self):
        return self.kurtosis

    def get_skewness(self):
        return self.skewness

    def get_entropy_freq(self):
        return self.freq_entropy

    def get_energy_freq(self):
        return self.freq_energy

    def get_correlation_freq(self):
        return self.freq_correlation

    def get_coefficient_of_variation_freq(self):
        return self.freq_cov

    def get_kurtosis_freq(self):
        return self.freq_kurtosis

    def get_skewness_freq(self):
        return self.freq_skewness
