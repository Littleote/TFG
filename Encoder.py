# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:22:19 2022

@author: david
"""

import numpy as np

class EncoderNone:
    def __init__(self):
        self.size = 1
    
    def encode(self, data):
        return data
    
    def decode(self, X):
        return X
    

class EncoderIgnore:
    def __init__(self, default=None):
        self.size = 0
        self.default = default
    
    def encode(self, data):
        return np.zeros((data.shape[0], 0))
    
    def decode(self, X):
        return np.full((X.shape[0], 1), self.default)
    
class EncoderOHE:
    def __init__(self, symbols):
        try:
            symbols = np.array(symbols, dtype=object)
            self.size = len(symbols)
            self.symbols = symbols
        except:
            if type(symbols) is int:
                self.size = symbols
                self.symbols = None
            else:
                raise ValueError("Invalid symbols type")
    
    def encode(self, data):
        if self.symbols is None:
            self.symbols = np.array(list(data.unique()) + [None for _ in range(self.size)], dtype=object)[:self.size] 
        return np.array([data == symbol for symbol in self.symbols], dtype=float).transpose()
    
    def decode(self, X):
        return np.reshape(self.symbols[np.argmax(X, 1)], (-1, 1))

class EncoderEquivalence:
    def __init__(self, symbols):
        symbols = np.array(symbols, dtype=object)
        self.size = 1
        self.symbols = len(symbols)
        values = np.arange(self.symbols)
        self.forward = {symb: val for symb, val in zip(symbols, values)}
        self.backward = {val: symb for symb, val in zip(symbols, values)}
    
    def encode(self, data):
        keys, val = np.unique(data, return_inverse=True)
        X = np.zeros(data.shape)
        for i, key in enumerate(keys):
            X[val == i] = self.forward.get(key, 0)
        return X
    
    def decode(self, X):
        X = X.astype(int)
        X[X < 0] = 0
        X[X >= self.symbols] = self.symbols - 1
        keys, val = np.unique(X, return_inverse=True)
        data = np.zeros(X.shape)
        for i, key in enumerate(keys):
            data[val == i] = self.backward.get(key)
        return X

def auto(data):
    if data.dtype == object:
        symbols = data.unique()
        use = np.sqrt(data.shape[0]) / symbols.shape[0]
        if use > 1:
            return EncoderOHE(list(symbols))
        else:
            return EncoderIgnore('Ignored')
    else:
        return EncoderNone()

def parse(name):
    name = name.lower()
    if name == 'none':
        return EncoderNone
    elif name == 'ignore':
        return EncoderIgnore
    elif name == 'ohe':
        return EncoderOHE
    raise ValueError(f"Invalid name ({name}) for encoder")