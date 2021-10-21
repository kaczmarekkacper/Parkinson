import configparser
import pandas as pd
# from Patient import Patient
from os import path
import random


class ImportData:
    @staticmethod
    def import_data():
        path_to_data = ImportData.get_path()
        demographics = ImportData.get_demographics()
        patients = []
        for index, row in demographics.iterrows():
            for i in range(1, 11):
                path_to_patient = path_to_data + row['ID'] + "_" + str(i).zfill(2) + ".txt"
                if path.exists(path_to_patient):
                    patient = Patient.Patient()
                    ImportData.fill_patient(patient, row)
                    data = pd.read_csv(path_to_patient, sep='\\t', header=0)
                    data.columns = ['Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
                                    'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Force_Left', 'Force_Right']
                    patient.sensor_readings = data
                    patient.walk_number = i
                    patients.append(patient)
        return patients

    @staticmethod
    def import_data_pandas():
        path_to_data = ImportData.get_path()
        demographics = ImportData.get_demographics()
        patients = []
        for index, row in demographics.iterrows():
            i = 1
            path_to_patient = path_to_data + row['ID'] + "_" + str(i).zfill(2) + ".txt"
            if path.exists(path_to_patient):
                data = pd.read_csv(path_to_patient, sep='\\t', header=0)
                data.columns = ['Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
                                'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Force_Left', 'Force_Right']
                data.insert(0, "ID", row['ID'], True)
                patients.append(data)
        patients = pd.concat(patients, axis=0, ignore_index=True)
        return patients

    @staticmethod
    def get_demographics():
        path_to_data = ImportData.get_path()
        path_to_file = path_to_data + "demographics.txt"
        demographics = pd.read_csv(path_to_file, sep='\\t', header=0)
        return demographics

    @staticmethod
    def get_path():
        config = configparser.ConfigParser()
        config_file = "config.ini"
        config.read(config_file)
        return config['DEFAULT']['path_to_data']

    @staticmethod
    def fill_patient(patient, row):
        patient.study_name = row['Study']
        patient.has_parkinson = (row['Group'] == 1)
        patient.subject_number = row['Subjnum']
        patient.is_male = (row['Gender'] == 1)
        patient.age = row['Age']
        patient.height = row['Height']
        patient.weight = row['Weight']
        patient.HoehnYahr = row['HoehnYahr']
        patient.UPDRS = row['UPDRS']
        patient.UPDRSM = row['UPDRSM']
        patient.TUAG = row['TUAG']
        speeds = []
        template = "Speed_"
        for i in [1, 2, 3, 4, 5, 6, 7, 10]:
            speeds.append(row[template + str(i).zfill(2)])
        patient.speeds = speeds
        patient.sampling_rate = 100