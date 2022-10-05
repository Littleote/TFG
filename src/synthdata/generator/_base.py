# -*- coding: utf-8 -*-
"""
Base Generator

@author: david
"""

import numpy as np

class BaseGenerator():
    def __init__(self, infer=False, **kwargs):
        self.infer = infer
        self.fit_args = dict()
        self.gen_args = dict()
        self.fill_args = dict()
        self.set_args(**kwargs)
        
    def set_args(self, **kwargs):
        fit = ('fit', self.set_fit_args)
        gen =  ('gen', self.set_generate_args)
        fill =  ('fill', self.set_fill_args)
        for code, f in (fit, gen, fill):
            f(**{k[len(code)+1:]: v for k, v in kwargs.items() if k[:len(code)+1] == (code + '_')})
        
    def set_fit_args(self, **kwargs):
        self.fit_args = self.fit_args | kwargs
        
    def set_generate_args(self, **kwargs):
        self.gen_args = self.gen_args | kwargs
        
    def set_fill_args(self, **kwargs):
        self.fill_args = self.fill_args | kwargs
    
    def infer(self, data):
        if self.infer:
            self._infer
        self.infer = False
        
    def _infer(self):
        NotImplementedError("This generator doesn't have inference from data functionality")
    
    def probabilities(self, X):
        NotImplementedError("This generator can't calculate the probability function")
    
    def loglikelihood(self, X):
        return np.sum(np.log(self.probabilities(X)))
    
    def fit(self, X, **kwargs):
        return self._fit(X, **self.fit_args | kwargs)
    
    def _fit(self, X):
        NotImplementedError("This generator doesn't have fit function")
    
    def generate(self, size, **kwargs):
        return self._generate(size, **self.gen_args | kwargs)
    
    def _generate(self, size):
        NotImplementedError("This generator doesn't have generate function")
    
    def fill(self, X, **kwargs):
        return self._fill(X, **self.fill_args | kwargs)
    
    def _fill(self, X):
        NotImplementedError("This generator doesn't have fill function")
    