from sklearn.naive_bayes import GaussianNB
from Alghoritms.Algorithm import Algorithm


class GaussNB(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = GaussianNB()

    def algorithm_name(self):
        return "GaussNB"
