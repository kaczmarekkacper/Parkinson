import pandas as pd


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
