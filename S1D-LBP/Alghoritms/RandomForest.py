from sklearn.ensemble import RandomForestClassifier
from Alghoritms.Algorithm import Algorithm


class RandomForest(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    def algorithm_name(self):
        return "RandomForest"
