
import numpy as np
from sklearn.metrics import confusion_matrix


class Algorithm:
    def __init__(self):
        self.input_size = None
        self.layers = None
        self.output_size = None
        self.layers_activation = None
        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

        self.model = None

    def train(self, train_set, lambdas):
        self.train_set = train_set
        self.train_labels = []

        train_inputs = []
        for example in train_set:
            for sensor in train_set[example]:
                self.train_labels.append(1 if self.train_set[example][sensor].has_parkinson else 0)
                train_input = []
                for lam in lambdas:
                    train_input.append((lam(train_set[example][sensor])))
                train_inputs.append(train_input)

        self.model.fit(np.array(train_inputs), np.array(self.train_labels))

    def test(self, test_set, lambdas):
        self.test_set = test_set
        self.test_labels = []

        test_inputs = []
        for example in test_set:
            for sensor in test_set[example]:
                self.test_labels.append(1 if self.test_set[example][sensor].has_parkinson else 0)
                test_input = []
                for lam in lambdas:
                    test_input.append(lam(self.test_set[example][sensor]))
                test_inputs.append(test_input)

        predict_labels = self.model.predict(np.array(test_inputs))

        correct = (np.array(self.test_labels) == predict_labels).sum()
        all = len(test_inputs)
        tn, fp, fn, tp = confusion_matrix(np.array(self.test_labels), predict_labels).ravel()
        accuracy = (tp + tn) / (tp+tn+fp+fn)
        sensitivity = tp / (tp+fn)
        specificity = tn / (fp+tn)

        print(f"{self.algorithm_name()}: {tp+tn} out of {all} = {round(correct/all*100, 2)}%")

        return accuracy, sensitivity, specificity

    def algorithm_name(self):
        return "Algorithm"
