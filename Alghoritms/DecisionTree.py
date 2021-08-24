from sklearn.tree import DecisionTreeClassifier
from Alghoritms.Algorithm import Algorithm


class DecisionTree(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = DecisionTreeClassifier()

    def algorithm_name(self):
        return "DecisionTree"
