from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np


class GaussNB:
    def __init__(self):
        self.input_size = None
        self.layers = None
        self.output_size = None
        self.layers_activation = None
        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

        self.model = GaussianNB()

    def train(self, train_set, lambdas):
        self.train_set = train_set
        self.train_labels = []

        train_inputs = []
        for example in train_set:
            self.train_labels.append(1 if example.has_parkinson else 0)
            train_input = []
            for lam in lambdas:
                train_input.append((lam(example)))
            train_inputs.append(train_input)

        self.model.fit(np.array(train_inputs), np.array(self.train_labels))

    def test(self, test_set, lambdas):
        self.test_set = test_set
        self.test_labels = []

        test_inputs = []
        for example in test_set:
            self.test_labels.append(1 if example.has_parkinson else 0)
            test_input = []
            for lam in lambdas:
                test_input.append(lam(example))
            test_inputs.append(test_input)

        predict_labels = self.model.predict(np.array(test_inputs))

        correct = (np.array(self.test_labels) == predict_labels).sum()
        all = len(test_inputs)

        print(f"GaussNB: {correct} out of {all} = {round(correct/all*100, 2)}%")
