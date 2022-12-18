# -*- coding: utf-8 -*-
"""
Base Generator

@author: david
"""

import numpy as np
import functools

def if_fitted(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.fitted
        return func(self, *args, **kwargs)
    return wrapper

def fit_method(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.fitted = True
        return func(self, *args, **kwargs)
    return wrapper

class Generator():
    def __init__(self):
        super().__init__()
        self.fitted = False
       
    @if_fitted
    def loglikelihood(self, X):
        return np.sum(np.log(self.probabilities(X) + 1e-300))
    