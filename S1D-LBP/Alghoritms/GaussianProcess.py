from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from Alghoritms.Algorithm import Algorithm


class GaussianProcess(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.model = GaussianProcessClassifier(1.0 * RBF(1.0))

    def algorithm_name(self):
        return "GaussianProcess"
