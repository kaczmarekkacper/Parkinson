from sklearn.model_selection import train_test_split
from ImportData import ImportData
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
# import cudf


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def prepare_patients(patients):
    for i in range(1, 9):
        patients[f'Diff{i}'] = patients.apply(lambda row: abs(row[f'L{i}'] - row[f'R{i}']), axis=1)



if __name__ == '__main__':
    config = load_config('config.ini')
    # patients = ImportData.ImportData.import_data_pandas()
    # # patients = cudf.DataFrame.from_pandas(patients_normal)
    # demographics = ImportData.ImportData.get_demographics()
    #
    # prepare_patients(patients)
    #
    # train_set_proportion = float(config['DEFAULT']['train_set_proportion'])
    # train_set, test_set = train_test_split(demographics, test_size=train_set_proportion, random_state=77)
    #
    # patients_train = pd.merge(patients, train_set['ID'], on='ID')
    # patients_test = pd.merge(patients, test_set['ID'], on='ID')
    #
    # train_ids = patients_train['ID'].unique()
    # test_ids = patients_test['ID'].unique()

    patients = pd.read_csv('Data/patients.csv')
    patients_train = pd.read_csv('Data/patients_train.csv')
    patients_test = pd.read_csv('Data/patients_test.csv')

    train_ids = patients_train['ID'].unique()
    test_ids = patients_test['ID'].unique()


    random_patient = patients_test[patients_test['ID'] == test_ids[0]].head(10)

    f, t, Zxx = signal.stft(random_patient['Diff2'], 100, nperseg=len(random_patient['Diff2']))
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=np.abs(Zxx.min()), vmax=np.abs(Zxx.max()), shading='gouraud')
    # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    print(t)
