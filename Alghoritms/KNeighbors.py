from sklearn.neighbors import KNeighborsClassifier
from Alghoritms.Algorithm import Algorithm


class KNeighbors(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = KNeighborsClassifier(3)

    def algorithm_name(self):
        return "KNeighbors"
