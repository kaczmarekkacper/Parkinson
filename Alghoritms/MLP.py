from sklearn.neural_network import MLPClassifier
from Alghoritms.Algorithm import Algorithm


class MLP(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = MLPClassifier(alpha=1, max_iter=10000)

    def algorithm_name(self):
        return "MLP"
