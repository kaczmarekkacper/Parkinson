from ImportData import ImportData
import random
from Alghoritms import NeuralNetwork


def split_data(patients, seed, proportion):
    train_set = []
    test_set = []

    with_parkinson = list(filter(lambda example: example.has_parkinson, patients))
    without_parkinson = list(filter(lambda example: not example.has_parkinson, patients))

    index = int(len(with_parkinson) * proportion)
    train_set += with_parkinson[:index]
    test_set += with_parkinson[index:]

    index = int(len(without_parkinson) * proportion)
    train_set += without_parkinson[:index]
    test_set += without_parkinson[index:]

    random.seed(seed)
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


if __name__ == '__main__':
    patients = ImportData.ImportData.import_data()

    seed = 7
    train_set_proportion = 0.7
    train_set, test_set = split_data(patients, seed, train_set_proportion)
    neural_network = NeuralNetwork.NeuralNetwork()
    lambdas = [(lambda e: e.get_min()), (lambda e: e.get_max()), (lambda e: e.get_mean()), (lambda e: e.get_mode()),
               (lambda e: e.get_kurtosis()), (lambda e: e.get_skewness())]
    neural_network.create_network(input_size=len(lambdas), layers=[100, 50], output_size=1,
                                  layers_activation=['relu', 'relu', 'sigmoid'])
    neural_network.train(train_set=train_set, lambdas=lambdas, number_of_epochs=1000)

    neural_network.test(test_set=test_set, lambdas=lambdas)
