import pandas as pd
import statistics


class Patient:
    def __init__(self):
        self.study_name = None
        self.has_parkinson = None
        self.prediction = None
        self.subject_number = None
        self.is_male = None
        self.age = None
        self.height = None
        self.weight = None
        self.HoehnYahr = None
        self.UPDRS = None
        self.UPDRSM = None
        self.TUAG = None
        self.speeds = None
        self.sampling_rate = None
        self.sensor_readings = pd.DataFrame()
        self.walk_number = None

    def get_min(self):
        return min(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5',
                                         'R6', 'R7', 'R8']].min())

    def get_max(self):
        return max(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5',
                                         'R6', 'R7', 'R8']].max())

    def get_mean(self):
        return statistics.mean(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
                                                     'R4', 'R5', 'R6', 'R7', 'R8']].mean())

    def get_mode(self):
        return statistics.mean(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
                                                     'R4', 'R5', 'R6', 'R7', 'R8']].kurtosis())

    #  energy

    def get_kurtosis(self):
        return statistics.mean(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
                                                     'R4', 'R5', 'R6', 'R7', 'R8']].kurtosis())

    def get_skewness(self):
        return statistics.mean(self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
                                                     'R4', 'R5', 'R6', 'R7', 'R8']].skew())
    def get_variance_all(self):
        var_dict = self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
                                                     'R4', 'R5', 'R6', 'R7', 'R8']].var()
        variance = [var_dict['L1'], var_dict['L2'], var_dict['L3'], var_dict['L4'], var_dict['L5'], var_dict['L6'],
                    var_dict['L7'], var_dict['L8'], var_dict['R1'], var_dict['R2'], var_dict['R3'], var_dict['R4'],
                    var_dict['R5'], var_dict['R6'], var_dict['R7'], var_dict['R8']]
        return variance

    # def get_entropy(self):
    #     return self.sensor_readings[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5',
    #     'R6', 'R7', 'R8']].corr()
