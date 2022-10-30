# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model

@author: david
"""

import numpy as np
from time import time as gettime

from ._base import BaseGenerator

class GMM(BaseGenerator):
    def __init__(self, k='auto', k_max=None, **kwargs):
        super().__init__(**kwargs)
        self.k_max = k_max
        if k == 'auto':
            self.k_mode = k
            self.k = None
        elif type(k) is int:
            self.k_mode = 'set'
            self.k = max(k, 1)
        else:
            raise ValueError("k must be an integer or 'auto'")
        
    def reset(self):
        self.weights = np.zeros(self.k) + 1 / self.k
        self.means = np.random.normal(size=(self.k, self.dim))
        self.covariances = np.array([np.identity(self.dim) for _ in range(self.k)])
        
    def set_k(self, k='auto', k_max=None):
        self.k_max = k_max
        if k == 'auto':
            self.k_mode = k
            self.k = None
        elif type(k) is int:
            self.k_mode = 'set'
            self.k = max(k, 1)
        else:
            raise ValueError("k must be an integer or 'auto'")
    
    def k_probability(self, X=None):
        if X is None:
            X = self.X
        assert X.shape[1] == self.dim, "Size mismatch"
        k_prob = []
        tol = 1e-6
        for weight, mean, covariance in \
            zip(self.weights, self.means, self.covariances):
                centered = X - mean
                _covariance = covariance + tol * np.identity(self.dim)
                determinant = np.linalg.det(_covariance)
                inverse = np.linalg.inv(_covariance)
                exponent = -np.sum(centered.dot(inverse) * centered, 1) / 2
                k_prob.append(weight * np.exp(exponent) / np.sqrt(determinant))
                    
        k_prob = np.array(k_prob).transpose() / np.power(2 * np.pi, self.dim / 2)
        return np.nan_to_num(k_prob)
        
    
    def probabilities(self, X):
        return np.sum(self.k_probability(X), 1)
    
    def iterate(self):
        probs = self.k_probability().transpose()
        resps = probs / np.maximum(np.sum(probs, 0), 1e-300)
        
        t_resps = np.maximum(np.sum(resps, 1), 1e-300)
        self.weights = t_resps / self.n
        self.means = np.transpose(resps.dot(self.X).transpose() / t_resps)
        self.covariances = np.array([
            (self.X - mean).transpose().dot(np.diag(resp).dot(self.X - mean)) / t_resp 
            for t_resp, resp, mean in 
                zip(t_resps, resps, self.means)
                ])
    
    def _fit(self, X, max_iter=1000, max_time=1, llh_tol=1e-3, n_attempts=3):
        self.X = X
        self.n, self.dim = X.shape
        if self.k_mode == 'auto':
            best_bic = None
            best_k = None
            gmm = GMM(fit_max_iter=max_iter, fit_max_time=max_time, \
                      fit_llh_tol=llh_tol, fit_n_attempts=n_attempts)
            k = 1
            k_lim = 2
            loop = k_lim if self.k_max is None else self.k_max
            while not loop == 0:
                loop -= 1
                gmm.set_k(k)
                gmm.fit(X)
                bic = k * (self.dim * (self.dim + 1) / 2 + self.dim + 1) * np.log(self.n) \
                    - 2 * gmm.loglikelihood(X)
                if best_bic is None or bic < best_bic:
                    best_bic = bic
                    best_k = k
                    if self.k_max is None:
                        loop = k_lim
                k += 1
            self.k = best_k
        best_llh = None
        for attempt in range(n_attempts):
            self.reset()
            loop = True
            iters = max_iter
            time = max_time + gettime()
            llh = self.loglikelihood(X)
            while loop:
                self.iterate()
                iters -= 1
                last_llh = llh
                llh = self.loglikelihood(X)
                loop &= iters > 0
                loop &= llh - last_llh > llh_tol
                loop &= time > gettime()
            if best_llh is None or llh > best_llh:
                best_llh = llh
                best_params = (self.weights, self.means, self.covariances)
        self.weights, self.means, self.covariances = best_params
            
    def _generate(self, size):
        ind, num = np.unique(np.random.choice(self.k, size, p=self.weights), return_counts=True)
        s = [
            np.random.multivariate_normal(self.means[i], self.covariances[i], n)
            for i, n in zip(ind, num)
            ]
        S = np.concatenate(s)
        return S
    
    def _fill(self, Y):
        assert Y.shape[1] == self.dim, "Size mismatch"
        for y in Y:
            bad = np.isnan(y)
            good = np.logical_not(bad)
            goods = np.sum(good)
            k_prob = []
            tol = 1e-6
            for weight, mean, covariance in \
                zip(self.weights, self.means, self.covariances):
                    centered = (y - mean)[good]
                    _covariance = covariance[good][:,good] + tol * np.identity(goods)
                    determinant = np.linalg.det(_covariance)
                    inverse = np.linalg.inv(_covariance)
                    exponent = (centered @ inverse @ centered) / 2
                    k_prob.append(weight * np.exp(-exponent) / np.sqrt(determinant))
            prob = np.array(k_prob)
            ind = np.random.choice(self.k, p=prob / np.sum(prob))
            covariance = self.covariances[ind]
            inv_subcov = np.linalg.inv(covariance[good][:,good] + tol * np.identity(goods))
            new_mean = self.means[ind][bad] + covariance[bad][:,good] @ inv_subcov @ (y - self.means[ind])[good]
            new_cov = covariance[bad][:,bad] - covariance[bad][:,good] @ inv_subcov @ covariance[good][:,bad]
            y[bad] = np.random.multivariate_normal(new_mean, new_cov)
        return Y
