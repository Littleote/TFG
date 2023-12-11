# -*- coding: utf-8 -*-
"""
Classes per codificar

@author: david
"""

import numpy as np

class Encoder:
    def encode(self, data):
        raise NotImplementedError()
    
    def decode(self, X):
        raise NotImplementedError()
    
    def toNan(self, data):
        raise NotImplementedError()

class EncoderNone(Encoder):
    def __init__(self):
        """
        Encoder that leaves the data unaltered when transforming.

        Returns
        -------
        Encoder.

        """
        self.size = 1
    
    def encode(self, data):
        return data
    
    def decode(self, X):
        return X
    
    def toNan(self, data):
        return np.isnan(data)

class EncoderDiscrete(Encoder):
    def __init__(self, minimum: 'int | None' = None, maximum: 'int | None' = None):
        """
        Encoder for discrete variables.

        Parameters
        ----------
        minimum : 'int | None', optional
            Minimum value of interval. If none, no minimum. The default is None.
        maximum : 'int | None', optional
            Maximum value of interval. If none, no maximum. The default is None.

        Returns
        -------
        Encoder.

        """
        self.size = 1
        self.has_min = minimum is not None
        self.has_max = maximum is not None
        self.min = np.nan if minimum is None else minimum
        self.max = np.nan if maximum is None else maximum
    
    def encode(self, data):
        return data.astype(float)
    
    def decode(self, X):
        X = np.round(X).astype(int)
        if self.has_min:
            X[X < self.min] = self.min
        if self.has_max:
            X[X > self.max] = self.max
        return X
    
    def toNan(self, data):
        return np.isnan(data)
    
class EncoderLimit(Encoder):
    def __init__(self, lower: 'float | None' = None, upper: 'float | None' = None, tails: 'bool' = True, influence: 'float' = 1):
        """
        Encoder for bounded continuous variables

        Parameters
        ----------
        lower : 'float | None', optional
            Lower bound of the interval. If none, unbound. The default is None.
        upper : 'float | None', optional
            Upper bound of the interval. If none, unbound. The default is None.
        tails : 'bool', optional
            Create tails at the extremes to make the interval open. The default is True.
        influence : 'float', optional
            Size of the tails when only one is present. The default is 1.

        Returns
        -------
        Encoder.

        """
        self.size = 1
        self.tails = tails
        assert lower is not None or upper is not None, "At least one limit must exist"
        self.invert = lower is None
        self.single = lower is None or upper is None
        self.a = upper if self.invert else lower 
        self.range = influence / 2 if self.single else (upper - lower) / 2
        self.sign = -1 if self.invert else 1
    
    def nan_out(self, X):
        tol = 0 if self.tails else 1e-6
        out = (-X if self.single else np.abs(X)) >= 1 + tol
        X[out] = np.nan
        return X
    
    def encode(self, data):
        X = self.nan_out(self.sign * (data - self.a) / self.range - 1)
        if self.tails and self.single:
            X = np.log(np.exp(X + 1) - 1)
        elif self.tails:
            X = np.log((1 + X) / (1 - X))
        return X
    
    def decode(self, X):
        if self.tails and self.single:
            X = np.log(np.exp(X) + 1) - 1
        elif self.tails:
            eX = np.exp(X)
            X = 2 * eX / (1 + eX) - 1
        else:
            X[X > 1] = 1
            X[X < -1] = -1
        data = self.a + self.sign * (X + 1) * self.range
        return data
    
    def toNan(self, data):
        return np.isnan(self.nan_out(self.sign * (data - self.a) / self.range - 1)) 

class EncoderIgnore(Encoder):
    def __init__(self, default=None):
        """
        Encoder for irrelevant variables

        Parameters
        ----------
        default : TYPE, optional
            Value to insert on the value ignored. The default is None.

        Returns
        -------
        Encoder.

        """
        self.size = 0
        self.default = default
    
    def encode(self, data):
        if self.default is None:
            if data.dtypes[0] == object:
                self.default = 'ignored'
            elif data.dtypes[0] == float:
                self.default = np.nan
            else:
                self.default = 0
        return np.zeros((data.shape[0], 0))
    
    def decode(self, X):
        return np.full((X.shape[0], 1), self.default)
    
    def toNan(self, data):
        return np.full(data.shape, False)
    
class EncoderOHE(Encoder):
    def __init__(self, symbols: 'list'):
        """
        One hot Encoder.

        Parameters
        ----------
        symbols : 'list'
            List of possible values.

        Returns
        -------
        Encoder.

        """
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
        X = np.array([data == symbol for symbol in self.symbols], dtype=float).transpose()
        X[np.sum(X, 2) == 0] = np.full(self.size, np.nan)
        return X
    
    def decode(self, X):
        return np.reshape(self.symbols[np.argmax(X, 1)], (-1, 1))
    
    def toNan(self, data):
        valid = np.vectorize(lambda x: x in self.symbols)
        return ~valid(data)

class EncoderScale(Encoder):
    def __init__(self, symbols: 'list'):
        """
        Encoder values as a scale.

        Parameters
        ----------
        symbols : 'list'
            Symbols of the scale in the order they appear.

        Returns
        -------
        Encoder.

        """
        symbols = np.array(symbols, dtype=object)
        self.size = 1
        self.symbols = len(symbols)
        assert self.symbols >= 1, "There must be at least one symbol"
        values = np.arange(self.symbols)
        self.forward = {symb: val for symb, val in zip(symbols, values)}
        self.backward = {val: symb for symb, val in zip(symbols, values)}
    
    def encode(self, data):
        keys, val = np.unique(data, return_inverse=True)
        X = np.zeros(data.shape)
        for i, key in enumerate(keys):
            X[val == i] = self.forward.get(key, np.nan)
        return X
    
    def decode(self, X):
        X = np.round(X).astype(int)
        X[X < 0] = 0
        X[X >= self.symbols] = self.symbols - 1
        X = np.round(X).astype(int)
        data = np.full(X.shape, self.backward[0], object)
        for val in range(1, self.symbols):
            data[X == val] = self.backward.get(val)
        return data
    
    def toNan(self, data):
        valid = np.vectorize(lambda x: x in self.forward.keys())
        return ~valid(data)
