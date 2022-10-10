# -*- coding: utf-8 -*-
"""
Data Class

@author: david
"""

import numpy as np
import pandas as pd

import synthdata.encoder as enc

class DataHub:
    def __init__(self):
        pass
    
    def load(self, data, encoders=dict(), method=None, preprocess="whitening"):
        assert len(data.shape) == 2, "data must be a 2d array of shape (n_samples, n_features)"
        
        self.data = data
        self.labels = list(data.columns)
        self.dtypes = data.dtypes.to_dict()
        self.encoders = {
            label: enc.auto(data[label])
            for label in self.labels
            } | encoders
        self.method = None
        self.preprocess = preprocess
        self.samples, self.features = data.shape
    
    def set_encoder(self, label, encoder):
        self.encoders[label] = encoder
        
    def set_method(self, method):
        self.method = method
        
    def transform(self, data=None):
        if data is None:
            data = self.data
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        X = np.zeros((data.shape[0], Xfeatures))
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            X[:,i:i + size] = self.encoders[label].encode(data[[label]])
            i += size
        self._fitprocess(X[~np.isnan(X).any(1)])
        return self._preprocess(X)
    
    def _fitprocess(self, X):
        if self.preprocess == 'normalize':
            self.mu = np.mean(X, 0)
            self.sigma = np.sqrt(np.var(X, 0))
        elif self.preprocess == 'whitening':
            self.mu = np.mean(X, 0)
            self.lambdas, self._U = np.linalg.eigh(np.cov(X.T))
            self.lambdas += 1e-6
        
    def _preprocess(self, X):
        if self.preprocess == 'normalize':
            return (X - self.mu) / self.sigma
        elif self.preprocess == 'whitening':
            return (X - self.mu) @ self._U @ np.diag(1 / np.sqrt(self.lambdas)) @ self._U.T
        else:
            return X
        
    def _postprocess(self, X):
        if self.preprocess == 'normalize':
            return X * self.sigma + self.mu
        elif self.preprocess == 'whitening':
            return X @ self._U @ np.diag(np.sqrt(self.lambdas)) @ self._U.T + self.mu
        else:
            return X
    
    def inv_transform(self, X):
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        X = self._postprocess(X)
        assert X.shape[1] == Xfeatures, f"X n_features ({X.shape[1]}) different from expected ({sum([enc.size for enc in self.encoders.values()])})"
        data = pd.DataFrame(columns=self.labels, dtype=object)
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            data[[label]] = self.encoders[label].decode(X[:,i:i + size])
            i += size
        return data.astype(self.dtypes)
    
    def generate(self, n_samples, method=None, target=None, preprocess=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = self.preprocess if preprocess is None else preprocess
        if target is None:
            X = self.transform()
            nans = np.isnan(X).any(axis=1)
            if nans.all():
                raise ValueError("All rows contain nan")
            X_ = X[np.logical_not(nans)]
            method.fit(X_, **model_args)
            return self.inv_transform(method.generate(n_samples))
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = enc.ignore(default=value)
                X = self.transform(self.data[self.data[target] == value])
                nans = np.isnan(X).any(axis=1)
                if nans.all():
                    raise ValueError("All rows for {target} = {value} contain nan")
                X_ = X[np.logical_not(nans)]
                method.fit(X_, **model_args)
                new_data.append(self.inv_transform(method.generate(n_samples)))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
            
    def fill(self, method=None, target=None, preprocess=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = self.preprocess if preprocess is None else preprocess
        if target is None:
            X = self.transform()
            nans = np.isnan(X).any(axis=1)
            if nans.all():
                raise ValueError("All rows contain nan")
            method.fit(X[np.logical_not(nans)], **model_args)
            X[nans] = method.fill(X[nans])
            return self.inv_transform(X)
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = enc.ignore(default=value)
                X = self.transform(self.data[self.data[target] == value])
                nans = np.isnan(X).any(axis=1)
                if nans.all():
                    raise ValueError("All rows for {target} = {value} contain nan")
                method.fit(X[np.logical_not(nans)], **model_args)
                X[nans] = method.fill(X[nans])
                new_data.append(self.inv_transform(X))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
        
    def extend(self, n_samples, method=None, target=None, preprocess=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = self.preprocess if preprocess is None else preprocess
        if method is None:
            method = self.method
        if target is None:
            if n_samples <= self.samples:
                return self.data.iloc[np.random.choice(self.samples, n_samples, False)]
            else:
                X = self.transform()
                nans = np.isnan(X).any(axis=1)
                if nans.all():
                    raise ValueError("All rows contain nan")
                X_ = X[np.logical_not(nans)]
                method.fit(X_, **model_args)
                Xnew = np.concatenate([
                    X, method.generate(n_samples - self.samples)
                    ])
                return self.inv_transform(Xnew)
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                subdata = self.data[self.data[target] == value]
                if n_samples <= len(subdata):
                    new_data.append(subdata.iloc[np.random.choice(len(subdata), n_samples, False)])
                else:
                    self.encoders[target] = enc.ignore(default=value)
                    X = self.transform(subdata)
                    nans = np.isnan(X).any(axis=1)
                    if nans.all():
                        raise ValueError("All rows for {target} = {value} contain nan")
                    X_ = X[np.logical_not(nans)]
                    method.fit(X_, **model_args)
                    Xnew = np.concatenate([
                        X, method.generate(n_samples - len(subdata))
                        ])
                    new_data.append(self.inv_transform(Xnew))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
