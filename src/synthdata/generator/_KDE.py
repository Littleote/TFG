# -*- coding: utf-8 -*-
"""
Kernel Density Estimation

@author: david
"""

import numpy as np

from ._base import Generator, if_fitted, fit_method

def constant_h(X):
    n, dim = X.shape
    variance = np.power(n - 1, -1 / dim)
    return np.sqrt(variance)

def optimize_h(X):
    n, dim = X.shape
    separator = int(n * 0.6)
    ind = np.random.choice(n, n, False)
    X_train = X[ind[:separator]]
    X_validation = X[ind[separator:]]
    values_h = dict()
    tolerance = 1e-2
    kde = KDE(0)
    kde.fit(X_train)
    def get_value(h):
        if h in values_h.keys():
            return values_h[h]
        kde.h = np.exp(h)
        value = kde.loglikelihood(X_validation)
        values_h[h] = value
        return value
    
    test_h = np.log(constant_h(X_train))
    if get_value(test_h) < get_value(test_h - 1):
        change = -1
    else:
        change = 1
    while get_value(test_h) < get_value(test_h + change):
        test_h = test_h + change
        change = 2 * change
        
    
    best_h = max(values_h, key=values_h.get)
    list_h = np.array(list(values_h.keys()))
    lower, middle, upper = max(list_h[list_h < best_h]), best_h, min(list_h[list_h > best_h])
    while lower + tolerance < upper:
        coeffs = lower, middle, upper
        values = [get_value(coeff) for coeff in coeffs]
        coeffs = [*coeffs, *coeffs]
        new_h = \
            sum([(coeffs[i+1]**2 - coeffs[i+2]**2) * values[i] for i in range(3)]) /\
            sum([(coeffs[i+1] - coeffs[i+2]) * values[i] for i in range(3)]) / 2
        if abs(new_h - middle) < tolerance:
            if upper - middle < middle - lower:
                new_h = middle - tolerance / 3
            else:
                new_h = middle + tolerance / 3
        if get_value(middle) < get_value(new_h):
            if middle < new_h:
                lower, middle, upper = middle, new_h, upper
            else:
                lower, middle, upper = lower, new_h, middle
        else:
            if middle < new_h:
                lower, middle, upper = lower, middle, new_h
            else:
                lower, middle, upper = new_h, middle, upper
    
    return np.exp(middle)

class KDE(Generator):
    def __init__(self, h: 'float | str | function' = 'tune'):
        super().__init__()
        if isinstance(h, str):
            if h.lower() in ['auto', 'tune']:
                self.h_generator = optimize_h
            elif h.lower() in ['constant', 'c']:
                self.h_generator = constant_h
            else:
                raise ValueError(f"Unrecognized h type ({h})")
        elif isinstance(h, int) or isinstance(h, float):
            self.h_generator = lambda X: h
        else:
            self.h_generator = h
    
    @fit_method
    def fit(self, X):
        self.X = X
        self.n, self.dim = X.shape
        self.h = self.h_generator(X)
       
    @if_fitted 
    def probabilities(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        dist = np.sum(np.square(
            np.reshape(self.X, (1, -1, self.dim))
            - np.reshape(X, (-1, 1, self.dim))
            ), 2)
        probs = np.sum(np.exp(-dist / 2 / self.h ** 2) / np.power(2 * np.pi * self.h ** 2, self.dim / 2), 1) / self.n
        return probs
      
    @if_fitted
    def generate(self, size):
        ind = np.random.choice(self.n, size)
        S = np.random.normal(self.X[ind], self.h)
        return S
    
    @if_fitted
    def fill(self, Y):
        assert Y.shape[1] == self.dim, "Size mismatch"
        diffs = np.reshape(self.X, (1, -1, self.dim)) - np.reshape(Y, (-1, 1, self.dim))
        dists = np.sum(np.square(np.nan_to_num(diffs)), 2)
        probs = np.exp(-dists / 2 / self.h ** 2)
        for y, prob in zip(Y, probs):
            bad = np.isnan(y)
            ind = np.random.choice(self.n, p=prob / np.sum(prob))
            y[bad] = np.random.normal(self.X[ind, bad], self.h)
        return Y
    