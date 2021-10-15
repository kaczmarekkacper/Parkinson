from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Alghoritms.Algorithm import Algorithm


class SVM(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    def algorithm_name(self):
        return "SVM"
