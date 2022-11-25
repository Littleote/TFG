# -*- coding: utf-8 -*-
"""
Kernel Density Estimation

@author: david
"""

import numpy as np

from ._base import BaseGenerator

class KDE(BaseGenerator):
    def __init__(self, var=lambda n, d: np.power(n, -1 / d), **kwargs):
        super().__init__(**kwargs)
        if type(var) == type(lambda:0):
            self.var = var
        else:
            self.var = lambda n, d: var
        
    def probabilities(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        dist = np.sum(np.square(
            np.reshape(self.X, (1, -1, self.dim))
            - np.reshape(X, (-1, 1, self.dim))
            ), 2)
        probs = np.sum(np.exp(-dist / 2 / self.var(self.n, self.dim)) / np.power(2 * np.pi * self.var(self.n, self.dim), self.dim / 2), 1) / self.n
        return probs
    
    def _fit(self, X):
        self.X = X
        self.n, self.dim = X.shape
            
    def _generate(self, size):
        ind = np.random.choice(self.n, size)
        S = np.random.normal(self.X[ind], np.sqrt(self.var(self.n, self.dim)))
        return S
    
    def _fill(self, Y):
        assert Y.shape[1] == self.dim, "Size mismatch"
        diffs = np.reshape(self.X, (1, -1, self.dim)) - np.reshape(Y, (-1, 1, self.dim))
        dists = np.sum(np.square(np.nan_to_num(diffs)), 2)
        probs = np.exp(-dists / 2 / self.var(self.n, self.dim))
        for y, prob in zip(Y, probs):
            bad = np.isnan(y)
            ind = np.random.choice(self.n, p=prob / np.sum(prob))
            y[bad] = np.random.normal(self.X[ind, bad], np.sqrt(self.var(self.n, self.dim)))
        return Y
    