# -*- coding: utf-8 -*-
"""
Transformer

@author: david
"""

import pandas as pd
import numpy as np

class Transformer():
    def __init__(self, normalize=True, remove_cov=True, whitening=False):
        super().__init__()
        self.encoders = dict()
        self.dtypes = dict()
        self.labels = []
        remove_cov = normalize and remove_cov
        whitening = remove_cov and whitening
        self.transforms = {
            "normalize": normalize,
            "remove_cov": remove_cov,
            "whitening": whitening
        }
    
    def add_encoders(self, encoders):
        self.encoders |= encoders
    
    def set_encoder(self, label, encoder):
        self.encoders[label] = encoder
        
    def get_info(self, data):
        self.labels = list(data.columns)
        self.dtypes = self.dtypes
        
    def copy_transformer(self):
        new_trans = Transformer()
        new_trans.encoders = self.encoders.copy()
        new_trans.labels = self.labels.copy()
        new_trans.dtypes = self.dtypes.copy()
        new_trans.transforms = self.transforms.copy()
        return new_trans
        
    def transform(self, data, refit=False, process=True):
        self.labels = list(data.columns)
        self.dtypes = data.dtypes.to_dict() | self.dtypes
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        X = np.zeros((data.shape[0], Xfeatures))
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            X[:,i:i + size] = self.encoders[label].encode(data[[label]])
            i += size
        if refit:
            self._fitprocess(X[~np.isnan(X).any(1)])
        if process:
            X = self._preprocess(X)
        return X
    
    def inv_transform(self, X, process=True):
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        if process:
            X = self._postprocess(X)
        assert X.shape[1] == Xfeatures, f"X n_features ({X.shape[1]}) different from expected ({sum([enc.size for enc in self.encoders.values()])})"
        data = pd.DataFrame(columns=self.labels, dtype=object)
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            data[[label]] = self.encoders[label].decode(X[:,i:i + size])
            i += size
        return data.astype(self.dtypes)
    
    def covariance(X, treshold):
        _, dim = X.shape
        nums = np.arange(dim)
        cov = np.cov(X.T)
        sigma = np.sqrt(np.var(X, 0))
        sigma_1 = 1 / (np.maximum(sigma, 0) + 1e-6)
        auto_cov = sigma_1 * (sigma_1 * cov).T
        conn = treshold < np.abs(auto_cov)
        conn[nums, nums] = True
        next_conn = conn @ conn
        while not np.all(conn == next_conn):
            conn = next_conn
            next_conn = conn @ conn
        relations = set([tuple(rel) for rel in conn])
        blocks = [nums[np.array(rel)] for rel in relations]
        signs = np.zeros(dim)
        for block in blocks:
            signs[block] = np.sign(auto_cov[block[0], block])
        return blocks, signs
    
    def _fitprocess(self, X):
        if self.transforms['normalize']:
            self.mu = np.mean(X, 0)
            self.sigma = np.sqrt(np.var(X, 0))
            self.sigma = np.maximum(self.sigma, 0) + 1e-6
            Y = (X - self.mu) / self.sigma
        if self.transforms['remove_cov']:
            self.blocks, self.sign = Transformer.covariance(Y, 0.975)
            self.dim_low, self.dim_high = len(self.blocks), Y.shape[1]
            Y = Y * self.sign
            Z = np.zeros((Y.shape[0], self.dim_low))
            for i, block in enumerate(self.blocks):
                Z[:,i] = np.nanmean(Y[:,block], 1)
        if self.transforms['whitening']:
            lambdas, self.U = np.linalg.eigh(np.cov(Z.T))
            self.s_lambdas = np.sqrt(np.maximum(lambdas, 0)) + 1e-6
            # W = Z @ self.U @ np.diag(1 / self.s_lambdas) @ self.U.T
        
    def _preprocess(self, X):
        if self.transforms['normalize']:
            X = (X - self.mu) / self.sigma
        if self.transforms['remove_cov']:
            X = X * self.sign
            Y = np.zeros((X.shape[0], self.dim_low))
            for i, block in enumerate(self.blocks):
                Y[:,i] = np.nanmean(X[:,block], 1)
            X = Y
        if self.transforms['whitening']:
            X = X @ self.U @ np.diag(1 / self.s_lambdas) @ self.U.T
        return X
        
    def _postprocess(self, X):
        if self.transforms['whitening']:
            X = X @ self.U @ np.diag(self.s_lambdas) @ self.U.T
        if self.transforms['remove_cov']:
            Y = np.zeros((X.shape[0], self.dim_high))
            for i, block in enumerate(self.blocks):
                Y[:,block] = X[:,[i]]
            X = Y * self.sign
        if self.transforms['normalize']:
            X = X * self.sigma + self.mu
        return X
    
    def deformation(self):
        deformation = 1
        if self.transforms['normalize']:
            deformation /= np.prod(self.sigma)
        if self.transforms['remove_cov']:
            ...
        if self.transforms['whitening']:
            deformation /= np.prod(self.s_lambdas)
        return deformation
    
    def run(self, data, FUN):
        nans = self.toNan(data)
        if nans.all():
            raise ValueError("All rows contain nan")
        X = self.transform(data[~nans], refit=True)
        FUN(X)
        return nans
    
    def toNan(self, data):
        nans = np.full(data.shape[0], False)
        for label in self.labels:
            nans |= self.encoders[label].toNan(data[label])
        return nans