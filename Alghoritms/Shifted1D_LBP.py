from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class Shifted1D_LBP:
    def __init__(self):
        self.no_neightbours = None
        self.no_central = None
        self.time = None
        self.signal_to_preprocess = None
        self.preprocessed_signal = None
        self.preprocessed_signal_numpy = None
        self.entropy = None
        self.energy = None
        self.correlation = None
        self.cov = None
        self.kurtosis = None
        self.skewness = None

    def shifting(self):
        self.preprocessed_signal = []
        for i in range(self.no_central, len(self.signal_to_preprocess) - self.no_neightbours + self.no_central - 1):
            central_point = []
            for idx in range(i - self.no_central, i + self.no_neightbours - self.no_central + 1):
                if idx == i:
                    continue
                t = self.signal_to_preprocess[idx] - self.signal_to_preprocess[i]
                central_point.append('1' if t >= 0 else '0')
            binary_number = ''.join(central_point)
            self.preprocessed_signal.append(self.convert_bin_to_int(binary_number))
        self.preprocessed_signal_numpy = np.array(self.preprocessed_signal)

    def convert_bin_to_int(self, str_number):
        return int(str_number, 2)

    def calculate_entropy(self):
        sum_squared_sqrt = np.sqrt(np.sum(np.square(self.preprocessed_signal_numpy)))
        entropy_function = lambda value: (value / sum_squared_sqrt) * np.log(value / sum_squared_sqrt)
        entropy = np.array([entropy_function(x) for x in self.preprocessed_signal_numpy])
        self.entropy = np.nansum(entropy)
        return self.entropy

    def calculate_energy(self):
        sum_squared = np.square(self.preprocessed_signal_numpy)
        energy = np.array([x * self.time[i + self.no_central] for i, x in enumerate(sum_squared)])
        self.energy = np.nansum(energy)
        return self.energy

    def calculate_correlation(self):
        signal_mean = np.mean(self.preprocessed_signal_numpy)
        signal_std = np.std(self.preprocessed_signal_numpy)
        correlation = np.array(
            [(i * x - signal_mean) / signal_std for i, x in enumerate(self.preprocessed_signal_numpy)])
        self.correlation = np.nansum(correlation)
        return self.correlation

    def calculate_coefficient_of_variation(self):
        signal_mean = np.mean(self.preprocessed_signal_numpy)
        signal_std = np.std(self.preprocessed_signal_numpy)
        self.cov = signal_std / signal_mean
        return self.cov

    def calculate_kurtosis(self):
        signal_mean = np.mean(self.preprocessed_signal_numpy)
        signal_std = np.std(self.preprocessed_signal_numpy)
        kurtosis_part = np.array([(i * x - signal_mean) ** 4 for i, x in enumerate(self.preprocessed_signal_numpy)])
        self.kurtosis = np.nansum(kurtosis_part) / self.preprocessed_signal_numpy.size / (signal_std ** 4)
        return self.kurtosis

    def calculate_skewness(self):
        signal_mean = np.mean(self.preprocessed_signal_numpy)
        signal_std = np.std(self.preprocessed_signal_numpy)
        skewness_part = np.array(
            [((i * x - signal_mean) / signal_std) ** 3 for i, x in enumerate(self.preprocessed_signal_numpy)])
        self.skewness = self.preprocessed_signal_numpy.size / (
                    (self.preprocessed_signal_numpy.size - 1) * (self.preprocessed_signal_numpy.size - 2)) * np.nansum(
            skewness_part)
        return self.skewness
