from sklearn.model_selection import train_test_split
import numpy as np


class AllTrue:
    def __init__(self):
        self.input_size = None
        self.layers = None
        self.output_size = None
        self.layers_activation = None
        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

    def train(self, train_set, lambdas):
        pass

    def test(self, test_set, lambdas):
        self.test_set = test_set
        self.test_labels = []

        test_inputs = []
        predict_labels = []
        for example in test_set:
            self.test_labels.append(1 if example.has_parkinson else 0)
            predict_labels.append(1)
            test_input = []
            for lam in lambdas:
                test_input.append(lam(example))
            test_inputs.append(test_input)

        correct = (np.array(self.test_labels) == predict_labels).sum()
        all = len(test_inputs)

        print(f"AllTrue: {correct} out of {all} = {round(correct/all*100, 2)}%")
