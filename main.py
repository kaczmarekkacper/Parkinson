from sklearn.model_selection import train_test_split

from ImportData import ImportData
from Alghoritms import GaussNB, SVM, KNeighbors, GaussianProcess, DecisionTree, RandomForest, MLP, QuadraticDiscriminant, AdaBoost
from tsfresh import extract_features
import configparser
import pandas as pd
from Alghoritms import Shifted1D_LBP
import math


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def check_if_has_parkinson(info, id):
    hoehn_yahr = info[info['ID'] == id]['HoehnYahr'].values[0]
    if math.isnan(hoehn_yahr):
        return False
    elif hoehn_yahr > 0:
        return True
    else:
        return False

def prepare_set(demographics, patients, ids, columns):
    prepared_set = {}
    for id in ids:
        person = {}
        for column in columns:
            alg = Shifted1D_LBP.Shifted1D_LBP()
            alg.no_central = 1
            alg.no_neightbours = 8
            alg.time = list(patients[patients['ID'] == id]['Time'])
            alg.signal_to_preprocess = list(patients[patients['ID'] == id][column])
            alg.shifting()
            alg.calculate_features()
            alg.has_parkinson = check_if_has_parkinson(demographics, id)
            person[column] = alg
        prepared_set[id] = person
    return prepared_set

if __name__ == '__main__':
    config = load_config('config.ini')
    patients = ImportData.ImportData.import_data_pandas()
    demographics = ImportData.ImportData.get_demographics()

    train_set_proportion = float(config['DEFAULT']['train_set_proportion'])
    train_set, test_set = train_test_split(demographics, test_size=train_set_proportion, random_state=77)

    patients_train = pd.merge(patients, train_set['ID'], on='ID')
    patients_test = pd.merge(patients, test_set['ID'], on='ID')

    train_ids = patients_train['ID'].unique()
    test_ids = patients_test['ID'].unique()

    columns = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
              'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Force_Left', 'Force_Right']

    train_set = prepare_set(demographics, patients_train, train_ids, columns)
    test_set = prepare_set(demographics, patients_test, test_ids, columns)


    lambdas = [lambda x: x.get_entropy(), lambda x: x.get_energy(), lambda x: x.get_correlation(), lambda x: x.get_coefficient_of_variation(), lambda x: x.get_kurtosis(), lambda x: x.get_skewness()]

    models = [GaussNB.GaussNB(), SVM.SVM(), KNeighbors.KNeighbors(), GaussianProcess.GaussianProcess(),
              DecisionTree.DecisionTree(),
              RandomForest.RandomForest(), MLP.MLP(), QuadraticDiscriminant.QuadraticDiscriminant(),
              AdaBoost.AdaBoost()]

    for model in models:
        model.train(train_set=train_set, lambdas=lambdas)
        model.test(test_set=test_set, lambdas=lambdas)

        
