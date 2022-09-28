# -*- coding: utf-8 -*-
"""
Variational AutoEncoder

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    vae = VAE()
    vae.model(X)
    
    syn = 1000
    S = vae.generate(syn)
    
    sdata = S * np.sqrt(var) + mean
    
    fig, ax = plt.subplots(1, 1)
    sp = ax.scatter(sdata[:,0], sdata[:,1], c='yellow', alpha=.2)
    sp = ax.scatter(data[:,0], data[:,1], c='red', alpha=.1)

    plt.show()