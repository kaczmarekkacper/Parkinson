from sklearn.model_selection import train_test_split

from ImportData import ImportData
from Alghoritms import GaussNB, KNeighbors, GaussianProcess, DecisionTree, RandomForest, MLP, QuadraticDiscriminant, AdaBoost, AllTrue
from tsfresh import extract_features
import configparser
import matplotlib.pyplot as plt
import pandas as pd
from Alghoritms import Shifted1D_LBP

def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


if __name__ == '__main__':
    config = load_config('config.ini')
    patients = ImportData.ImportData.import_data_pandas()
    print(patients.head())
    demographics = ImportData.ImportData.get_demographics()
    print(demographics.head())

    train_set_proportion = float(config['DEFAULT']['train_set_proportion'])
    train_set, test_set = train_test_split(demographics, test_size=train_set_proportion, random_state=77)
    print(train_set.head())
    print(test_set.head())

    patients_train = pd.merge(patients, train_set['ID'], on='ID')
    patients_test = pd.merge(patients, test_set['ID'], on='ID')

    alg = Shifted1D_LBP.Shifted1D_LBP()
    alg.no_central = 2
    alg.no_neightbours = 8
    alg.time = list(patients_train[patients_train['ID'] == 'SiPt20']['Time'])
    alg.signal_to_preprocess = list(patients_train[patients_train['ID'] == 'SiPt20']['Force_Left'])

    alg.shifting()

    alg.calculate_entropy()
    alg.calculate_energy()
    alg.calculate_correlation()
    alg.calculate_coefficient_of_variation()
    alg.calculate_kurtosis()

    plt.plot(patients_train[patients_train['ID'] == 'SiPt20']['Time'][0:-9],
             alg.preprocessed_signal)
    plt.title('SiPt20 Force_Left shifted')
    plt.show()

    plt.plot(patients_train[patients_train['ID'] == 'SiPt20']['Time'],
             patients_train[patients_train['ID'] == 'SiPt20']['Force_Left'])
    plt.title('SiPt20 Force_Left')
    plt.show()


        
