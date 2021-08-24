from sklearn.ensemble import AdaBoostClassifier
from Alghoritms.Algorithm import Algorithm


class AdaBoost(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = AdaBoostClassifier()

    def algorithm_name(self):
        return "AdaBoost"
