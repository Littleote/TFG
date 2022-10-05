# -*- coding: utf-8 -*-
"""
Variational AutoEncoder

@author: david
"""

import numpy as np

class VAE:
    def __init__(self):
        pass
        # TODO
        
    def probabilities(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        # TODO
        
    def model(self, X):
        self.X = X
        self.n, self.dim = X.shape
        # TODO
            
    def generate(self, size):
        ind = np.random.choice(self.n, size)
        S = np.random.normal(self.X[ind], self.var(self.n))
        # TODO
        return S