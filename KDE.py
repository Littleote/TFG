# -*- coding: utf-8 -*-
"""
Kernel Density Estimation

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    n = 100
    
    phase = np.linspace(0, 2 * np.pi, n)
    noise = [np.random.normal(size=n) for _ in range(2)]
    data = np.array([
            np.sin(phase + noise[0]) + .1 * np.sin(phase + noise[1]),
            np.cos(phase + noise[0]) + .1 * np.cos(phase + noise[1])
        ]).transpose() * 10 + np.array([1, 5])
    
    mean = np.mean(data, 0)
    var = np.var(data, 0)
    
    X = (data - mean) / np.sqrt(var)
    
    kde = KDE()
    kde.model(X)
    
    syn = 1000
    fdata = np.random.normal(mean, np.sqrt(var) / 2, size=(syn, 2)) + np.full((syn, 2), [np.nan, 0])
    F = kde.fill((fdata - mean) / np.sqrt(var))
    S = kde.generate(syn)
    
    sdata = S * np.sqrt(var) + mean
    fdata = F * np.sqrt(var) + mean
    
    fig, ax = plt.subplots(1, 1)
    sp = ax.scatter(sdata[:,0], sdata[:,1], c='yellow', alpha=.2)
    sp = ax.scatter(fdata[:,0], fdata[:,1], c='green', alpha=.1)
    sp = ax.scatter(data[:,0], data[:,1], c='red', alpha=.2)

    plt.show()