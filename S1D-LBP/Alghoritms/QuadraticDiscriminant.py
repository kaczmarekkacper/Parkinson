from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Alghoritms.Algorithm import Algorithm


class QuadraticDiscriminant(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = QuadraticDiscriminantAnalysis()

    def algorithm_name(self):
        return "QuadraticDiscriminant"
