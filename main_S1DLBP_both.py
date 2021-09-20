from sklearn.model_selection import train_test_split

from ImportData import ImportData
from Alghoritms import GaussNB, SVM, KNeighbors, GaussianProcess, DecisionTree, RandomForest, MLP, \
    QuadraticDiscriminant, AdaBoost
from tsfresh import extract_features
import configparser
import pandas as pd
from Alghoritms import Shifted1D_LBP
import math


# def make_results(train_set, test_set):


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


def prepare_set(demographics, patients, ids, columns, no_central):
    prepared_set = {}
    for id in ids:
        person = {}
        for column in columns:
            alg = Shifted1D_LBP.Shifted1D_LBP()
            alg.no_central = no_central
            alg.no_neightbours = 8
            alg.time = list(patients[patients['ID'] == id]['Time'])
            alg.signal_to_preprocess = list(patients[patients['ID'] == id][column])
            alg.shifting()
            alg.calculate_features()
            alg.shifting_freq()
            alg.calculate_features_freq()
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

    lambdas = [lambda x: x.get_entropy(), lambda x: x.get_energy(), lambda x: x.get_correlation(),
               lambda x: x.get_coefficient_of_variation(), lambda x: x.get_kurtosis(), lambda x: x.get_skewness(),
               lambda x: x.get_entropy_freq(), lambda x: x.get_energy_freq(), lambda x: x.get_correlation_freq(),
               lambda x: x.get_coefficient_of_variation_freq(), lambda x: x.get_kurtosis_freq(),
               lambda x: x.get_skewness_freq()
               ]

    data = {}
    for no_central in range(9):
        data[f"({no_central}, {8 - no_central})"] = {}
        print(no_central)
        train_set = prepare_set(demographics, patients_train, train_ids, columns, no_central)
        test_set = prepare_set(demographics, patients_test, test_ids, columns, no_central)
        models = [GaussNB.GaussNB(), SVM.SVM(), KNeighbors.KNeighbors(), GaussianProcess.GaussianProcess(),
                  DecisionTree.DecisionTree(),
                  RandomForest.RandomForest(), MLP.MLP(), QuadraticDiscriminant.QuadraticDiscriminant(),
                  AdaBoost.AdaBoost()]
        for model in models:
            print(model.algorithm_name())
            model.train(train_set=train_set, lambdas=lambdas)
            accuracy, sensitivity, specificity = model.test(test_set=test_set, lambdas=lambdas)
            data[f"({no_central}, {8 - no_central})"][f"{model.algorithm_name()}"] = accuracy #[accuracy, sensitivity, specificity]

    df = pd.DataFrame(data)

    df.to_csv('results_both.csv')
