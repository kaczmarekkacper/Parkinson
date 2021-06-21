from ImportData import ImportData
from Alghoritms import NeuralNetwork
from tsfresh import extract_features


if __name__ == '__main__':
    patients = ImportData.ImportData.import_data_pandas()

    df_features = extract_features(patients, column_id="ID", column_sort="Time")
    print(df_features)
    # seed = 7
    # train_set_proportion = 0.7
    # train_set, test_set = ImportData.ImportData.split_data(patients, seed, train_set_proportion)
    # neural_network = NeuralNetwork.NeuralNetwork()
    #
    # # neural network mix
    # # lambdas = [(lambda e: e.get_min()), (lambda e: e.get_max()), (lambda e: e.get_mean()), (lambda e: e.get_mode()),
    # #            (lambda e: e.get_kurtosis()), (lambda e: e.get_skewness())]
    # # neural_network.create_network(input_size=len(lambdas), layers=[100, 50], output_size=1,
    # #                               layers_activation=['relu', 'relu', 'sigmoid'])
    # # neural_network.train(train_set=train_set, lambdas=lambdas, number_of_epochs=1000)
    # #
    # # neural_network.test(test_set=test_set, lambdas=lambdas)
    #
    # # neural network variance
    # lambdas = [(lambda e: e.get_variance_all())]
    # neural_network.create_network(input_size=len(lambdas)*16, layers=[16, 16, 16], output_size=1,
    #                               layers_activation=['relu', 'relu', 'sigmoid'])
    # neural_network.train(train_set=train_set, lambdas=lambdas, number_of_epochs=10000)
    #
    # neural_network.test(test_set=test_set, lambdas=lambdas)

