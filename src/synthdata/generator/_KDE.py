# -*- coding: utf-8 -*-
"""
Kernel Density Estimation

@author: david
"""

import numpy as np

class KDE:
    def __init__(self, var=lambda n: 1 / n):
        self.var = var
        
    def probabilities(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        dist = np.sum(np.square(
            np.reshape(self.X, (1, -1, self.dim))
            - np.reshape(X, (-1, 1, self.dim))
            ), 2)
        probs = np.sum(np.exp(-dist / 2 / self.var(self.n)) / np.sqrt(2 * np.pi), 1) / self.n
        return probs
    
    def loglikelihood(self, X):
        return np.sum(np.log(self.probabilities(X)))
        
    def model(self, X):
        self.X = X
        self.n, self.dim = X.shape
            
    def generate(self, size):
        ind = np.random.choice(self.n, size)
        S = np.random.normal(self.X[ind], np.sqrt(self.var(self.n)))
        return S
    
    def fill(self, Y):
        assert Y.shape[1] == self.dim, "Size mismatch"
        diffs = np.reshape(self.X, (1, -1, self.dim)) - np.reshape(Y, (-1, 1, self.dim))
        dists = np.sum(np.square(np.nan_to_num(diffs)), 2)
        probs = np.exp(-dists / 2 / self.var(self.n))
        for y, prob in zip(Y, probs):
            bad = np.isnan(y)
            ind = np.random.choice(self.n, p=prob / np.sum(prob))
            y[bad] = np.random.normal(self.X[ind, bad], np.sqrt(self.var(self.n)))
        return Y
    