from ._base import Generator, if_fitted, fit_method
from ._GMM import GMM
from ._KDE import KDE

class AutoSelection(Generator):
    def __init__(self):
        super().__init__()
        self.generator = None

    @fit_method
    def fit(self, X):
        self.X = X
        self.n, self.dim = X.shape
        if 200 <= self.n <= 2e3:
            self.generator = GMM()
        else:
            self.generator = KDE()
        self.generator.fit(X)
       
    @if_fitted 
    def probabilities(self, X):
        return self.generator.probabilities(X)
      
    @if_fitted
    def generate(self, size):
        return self.generator.generate(size)
    
    @if_fitted
    def fill(self, Y):
        return self.generator.fill(Y)