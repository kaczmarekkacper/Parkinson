from ImportData import ImportData
from Alghoritms import GaussNB, SVM, KNeighbors, GaussianProcess, DecisionTree, RandomForest, MLP, QuadraticDiscriminant, AdaBoost, AllTrue
from tsfresh import extract_features


if __name__ == '__main__':
    patients = ImportData.ImportData.import_data()
    # print(patients.head())
    # demographics = ImportData.ImportData.get_demographics()
    # print(demographics.head())

    # GaPt03 = patients[patients['ID'] == 'GaPt03']

    # print(GaPt03.head())


    # df_features = extract_features(GaPt03, column_id="ID", column_sort="Time")
    # print(df_features)
    seed = 7
    train_set_proportion = 0.7
    train_set, test_set = ImportData.ImportData.split_data(patients, seed, train_set_proportion)

    # neural network mix
    lambdas = [(lambda e: e.get_min()), (lambda e: e.get_max()), (lambda e: e.get_mean()), (lambda e: e.get_mode()),
               (lambda e: e.get_kurtosis()), (lambda e: e.get_skewness()), (lambda e: e.get_energy()), (lambda e: e.get_coefficient_of_variation())]

    models = [GaussNB.GaussNB(), SVM.SVM(), KNeighbors.KNeighbors(), GaussianProcess.GaussianProcess(), DecisionTree.DecisionTree(),
    RandomForest.RandomForest(), MLP.MLP(), QuadraticDiscriminant.QuadraticDiscriminant(), AdaBoost.AdaBoost(), AllTrue.AllTrue()]
    # models =[AllTrue.AllTrue()]
    for model in models:
        model.train(train_set=train_set, lambdas=lambdas)
        model.test(test_set=test_set, lambdas=lambdas)
        
