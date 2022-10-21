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
    
    def load(self, data, encoders=dict(), method=None):
        assert len(data.shape) == 2, "data must be a 2d array of shape (n_samples, n_features)"
        
        self.data = data
        self.labels = list(data.columns)
        self.dtypes = data.dtypes.to_dict()
        self.encoders = {
            label: enc.auto(data[label])
            for label in self.labels
            } | encoders
        self.method = None
        self.samples, self.features = data.shape
    
    def set_encoder(self, label, encoder):
        self.encoders[label] = encoder
        
    def set_method(self, method):
        self.method = method
        
    def transform(self, data=None, preprocess=None, refit=False):
        preprocess = self.preprocess if preprocess is None else preprocess
        if data is None:
            data = self.data
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        X = np.zeros((data.shape[0], Xfeatures))
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            X[:,i:i + size] = self.encoders[label].encode(data[[label]])
            i += size
        if refit:
            self._fitprocess(X[~np.isnan(X).any(1)], preprocess)
        return self._preprocess(X, preprocess)
    
    def inv_transform(self, X, preprocess=None):
        preprocess = self.preprocess if preprocess is None else preprocess
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        X = self._postprocess(X, preprocess)
        assert X.shape[1] == Xfeatures, f"X n_features ({X.shape[1]}) different from expected ({sum([enc.size for enc in self.encoders.values()])})"
        data = pd.DataFrame(columns=self.labels, dtype=object)
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            data[[label]] = self.encoders[label].decode(X[:,i:i + size])
            i += size
        return data.astype(self.dtypes)
    
    def _fitprocess(self, X, preprocess):
        if preprocess == 'normalize':
            self.mu = np.mean(X, 0)
            self.sigma = np.sqrt(np.var(X, 0))
        elif preprocess == 'whitening':
            self.mu = np.mean(X, 0)
            self.lambdas, self._U = np.linalg.eigh(np.cov(X.T))
            self.lambdas += 1e-6
        
    def _preprocess(self, X, preprocess):
        if preprocess == 'normalize':
            return (X - self.mu) / self.sigma
        elif preprocess == 'whitening':
            return (X - self.mu) @ self._U @ np.diag(1 / np.sqrt(self.lambdas)) @ self._U.T
        else:
            return X
        
    def _postprocess(self, X, preprocess):
        if preprocess == 'normalize':
            return X * self.sigma + self.mu
        elif preprocess == 'whitening':
            return X @ self._U @ np.diag(np.sqrt(self.lambdas)) @ self._U.T + self.mu
        else:
            return X
    
    def kfold_validation(self, n_samples=None, folds=None, method=None, validation='loglikelihood', target=None, return_fit=False):
        method = self.method if method is None else method
        self.preprocess = 'normalize'
        if n_samples is None and folds is None:
            raise ValueError("No value specified for kfold validation")
        def sft(n_samples, folds, total):
            if folds is None:
                assert total >= 2 * n_samples, "n_samples to big for the data"
                folds = total // n_samples
                return n_samples, folds, n_samples * folds
            elif n_samples is None:
                assert folds >= 2, "folds must be at least 2"
                n_samples = total // folds
                return n_samples, folds, folds * n_samples
            else:
                assert total >= 2 * n_samples, "n_samples to big for the data"
                folds = min(folds, total // n_samples)
                return n_samples, folds, n_samples * folds
            
        if target is None:
            s, f, t = sft(n_samples, folds, self.samples)
            sampling = np.reshape(np.random.choice(self.samples, t, False), (f, -1))
            value = 0
            selfvalue = 0
            for i in range(f):
                train = self.data.iloc[sampling[i]]
                test = self.data.iloc[sampling[(i + 1) % f]]
                self.run(train, method)
                if validation == 'loglikelihood':
                    selfvalue += method.loglikelihood(self.transform(train)) / s
                    value += method.loglikelihood(self.transform(test)) / s
                else:
                    Xgen = self.transform(self.inv_transform(method.generate(s)))
                    Xtrain = self.transform(train)
                    Xtest = self.transform(test)
                    selfvalue += validation(Xtrain, Xgen)
                    value += validation(Xtest, Xgen)
            selfvalue /= f
            value /= f
            return (value, selfvalue) if return_fit else value
        else:
            pass
    
    def run(self, data, method, **method_args):
        nans = self.toNan(data)
        if nans.all():
            raise ValueError("All rows contain nan")
        X = self.transform(data[~nans], refit=True)
        method.fit(X, **method_args)
        return nans
    
    def toNan(self, data):
        nans = np.full(data.shape[0], False)
        for label in self.labels:
            nans |= self.encoders[label].toNan(data[label])
        return nans
    
    def generate(self, n_samples, method=None, target=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = 'whitening'
        if target is None:
            self.run(self.data, method, **model_args)
            return self.inv_transform(method.generate(n_samples))
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = enc.ignore(default=value)
                self.run(self.data[self.data[target] == value], method, **model_args)
                new_data.append(self.inv_transform(method.generate(n_samples)))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
            
    def fill(self, method=None, target=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = 'normalize'
        if target is None:
            new_data = self.data.copy()
            nans = self.run(new_data, method, **model_args)
            new_data.iloc[nans] = self.inv_transform(
                method.fill(self.transform(new_data[nans]))
            )
            return new_data
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = enc.ignore(default=value)
                subdata = self.data[self.data[target] == value].copy()
                nans = self.run(subdata, method, **model_args)
                subdata.iloc[nans] = self.inv_transform(
                    method.fill(self.transform(subdata[nans]))
                )
                new_data.append(subdata)
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
        
    def extend(self, n_samples, max_samples='n_samples', method=None, target=None, **model_args):
        method = self.method if method is None else method
        self.preprocess = 'whitening'
        if max_samples == 'n_samples':
            max_samples = n_samples
        elif max_samples is None:
            max_samples = self.samples
        else:
            max_samples = max(max_samples, n_samples)
            
        if method is None:
            method = self.method
        if target is None:
            original_data = self.data.copy()
            if n_samples <= self.samples:
                return original_data.iloc[np.random.choice(self.samples, min(max_samples, self.samples), False)]
            else:
                self.run(self.data, method, **model_args)
                new_data = self.inv_transform(method.generate(n_samples - self.samples))
                return pd.concat([original_data, new_data])
        else:
            target_encoder = self.encoders[target]
            original_data = []
            new_data = []
            for value in self.data[target].unique():
                subdata = self.data[self.data[target] == value]
                subsample = len(subdata)
                if n_samples <= subsample:
                    original_data.append(subdata.iloc[np.random.choice(subsample, min(max_samples, subsample), False)])
                else:
                    self.encoders[target] = enc.ignore(default=value)
                    original_data.append(subdata)
                    self.run(subdata, method, **model_args)
                    Xnew = method.generate(n_samples - subsample)
                    new_data.append(self.inv_transform(Xnew))
            self.encoders[target] = target_encoder
            return pd.concat(original_data + new_data)
