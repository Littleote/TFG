# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model

@author: david
"""

import numpy as np
from time import time as gettime

from ._base import Generator, if_fitted, fit_method

PHI = (np.sqrt(5) - 1) / 2

class Bayesian():
    def __init__(self, X):
        self.X = X
        self.n, self.dim = X.shape
        
    def data(self):
        return self.X
    
    def value(self, cls):
        return cls._degrees_freedom() * np.log(self.n) - 2 * cls.loglikelihood(self.X)

class Akaike():
    def __init__(self, X):
        self.X = X
        self.n, self.dim = X.shape
        
    def data(self):
        return self.X
    
    def value(self, cls):
        return 2 * cls._degrees_freedom() - 2 * cls.loglikelihood(self.X)

class CrossValidation():
    def __init__(self, X):
        self.X = X
        self.n, self.dim = X.shape
        self.ind = np.random.choice(self.n, self.n, False)
        self.separator = int(self.n * 0.6)
        
    def data(self):
        return self.X[self.ind[:self.separator]]
    
    def value(self, cls):
        return - cls.loglikelihood(self.X[self.ind[self.separator:]])

def parse_criterion(criterion, X):
    if not isinstance(criterion, str):
        raise ValueError('Criterion must be a string')
    elif criterion.lower() in ['bayesian', 'b', 'bic']:
        return Bayesian(X)
    elif criterion.lower() in ['akaike', 'a', 'aic']:
        return Akaike(X)
    elif criterion.lower() in ['cross-validation', 'cv']:
        return CrossValidation(X)
    raise ValueError(f'{criterion} is not a recognized criterion, use AIC, CV or BIC instead')

class GMM(Generator):
    def __init__(self, k: 'int | None' = None, k_max: 'int | None' = None, multivariate: 'bool' = True,
                 iterations_limit: 'int' = 1000, time_limit: 'float' = 1,
                 llh_tolerance: 'float' = 1e-3, attempts: 'int' = 3, criterion: 'str' = 'BIC'):
        super().__init__()
        if k is None:
            self.k_mode = 'search'
            self.k_max = k_max
            self.k = None
        elif isinstance(k, int):
            self.k_mode = 'set'
            self.k_max = None
            self.k = max(k, 1)
        else:
            raise ValueError("k must be an integer or None")
        self.multivariate = multivariate
        self.iterations_limit = iterations_limit
        self.time_limit = time_limit
        self.llh_tolerance = llh_tolerance
        self.attempts = attempts
        self.criterion = criterion
    
    @fit_method
    def fit(self, X):
        self.X = X
        self.n, self.dim = X.shape
        if self.k_mode == 'search':
            # Search optimal K value
            criterion = parse_criterion(self.criterion, X)
            X_search = criterion.data()
            gmm = GMM(
                multivariate=self.multivariate, \
                iterations_limit=self.iterations_limit, \
                time_limit=self.time_limit, \
                llh_tolerance=self.llh_tolerance, \
                attempts=self.attempts
            )
            if self.k_max is None:
                criterion_values = dict()
                best_value = np.Inf
                k = 1
                loop = True
                while loop:
                    loop = False
                    k = 2 * k
                    gmm._set_k(k)
                    gmm.fit(X_search)
                    criterion_values[k] = criterion.value(gmm)
                    if criterion_values[k] < best_value:
                        best_value = criterion_values[k]
                        best_k = k
                        loop = True
                best_k = gmm._optimize_in(X_search, criterion, \
                                          precision=int(np.sqrt(best_k)), \
                                          min_k=int(best_k / 2), max_k=int(best_k * 2), \
                                          criterion_values=criterion_values)
            else:
                best_k = gmm._optimize_in(X_search, criterion, \
                                          min_k=1, max_k=self.k_max)
            self.k = best_k
          
        # Run Expectation-Maximization algorithm
        best_llh = None
        for attempt in range(self.attempts):
            self._reset()
            loop = True
            iteration_left = self.iterations_limit
            time_limit = self.time_limit + gettime()
            llh = self.loglikelihood(X)
            while loop:
                self._iterate(self.X)
                self._step()
                    
                iteration_left -= 1
                last_llh = llh
                llh = self.loglikelihood(X)
                
                loop &= iteration_left > 0
                loop &= llh - last_llh > self.llh_tolerance
                loop &= time_limit > gettime()
                
            if not best_llh or llh > best_llh:
                best_llh = llh
                best_params = (self.weights, self.means, self.covariances, self.inverses, self.determinants)
        self.weights, self.means, self.covariances, self.inverses, self.determinants = best_params
    
    @if_fitted
    def probabilities(self, X):
        return np.sum(self._k_probability(X), 1)
    
    @if_fitted   
    def generate(self, size):
        ind, num = np.unique(np.random.choice(self.k, size, p=self.weights), return_counts=True)
        s = [
            np.random.multivariate_normal(self.means[i], self.covariances[i], n)
            for i, n in zip(ind, num)
            ]
        S = np.concatenate(s)
        return S
    
    @if_fitted
    def fill(self, Y):
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
        
    def _reset(self):
        self.weights = np.zeros(self.k) + 1 / self.k
        self.means = np.random.normal(size=(self.k, self.dim))
        self.covariances = np.array([np.identity(self.dim) for _ in range(self.k)])
        self.inverses = np.array([np.identity(self.dim) for _ in range(self.k)])
        self.determinants = np.ones(self.k)
        
    def _set_k(self, k):
        self.k_mode = 'set'
        self.k_max = None
        self.k = max(k, 1)
    
    def _k_probability(self, X):
        assert X.shape[1] == self.dim, "Size mismatch"
        k_prob = []
        for weight, mean, inverse, determinant in \
            zip(self.weights, self.means, self.inverses, self.determinants):
                centered = X - mean
                exponent = -np.sum(centered.dot(inverse) * centered, 1) / 2
                k_prob.append(weight * np.exp(exponent) / np.sqrt(determinant))
                    
        k_prob = np.array(k_prob).transpose() / np.power(2 * np.pi, self.dim / 2)
        return np.nan_to_num(k_prob)
    
    def _iterate(self, X):
        # Expectation
        probs = self._k_probability(X).transpose()
        resps = probs / np.maximum(np.sum(probs, 0), 1e-300)
        t_resps = np.maximum(np.sum(resps, 1), 1e-300)
        
        # Maximization
        self.new_weights = t_resps / self.n
        self.new_weights /= np.sum(self.new_weights)
        self.new_means = np.transpose(resps.dot(X).T / t_resps)
        self.new_covariances = np.array([
            ((X - mean).T * resp) @ (X - mean) / t_resp 
            for t_resp, resp, mean in 
                zip(t_resps, resps, self.new_means)
                ])
    
    def _step(self):
        self.weights = self.new_weights
        self.means = self.new_means
        diag = np.arange(self.dim)
        if self.multivariate:
            self.covariances = self.new_covariances
            self.covariances[:, diag, diag] += 1e-6
        else:
            self.covariances[:, diag, diag] = self.new_covariances[:, diag, diag] + 1e-6
        for i in range(self.k):
            if self.multivariate:
                self.inverses[i] = np.linalg.inv(self.covariances[i])
                self.determinants[i] = np.linalg.det(self.covariances[i])
            else:
                self.inverses[i, diag, diag] = 1 / self.covariances[i].diagonal()
                self.determinants[i] = np.prod(self.covariances[i].diagonal())
    
    def _degrees_freedom(self):
        if self.multivariate:
            return self.k * (self.dim ** 2 + 3 * self.dim + 2) / 2 - 1
        else:
            return self.k * (2 * self.dim + 1) - 1
    
    def _optimize_in(self, X, criterion, max_k, min_k=1, precision=1, criterion_values = dict()):
        def get_value(k):
            if k in criterion_values.keys():
                return criterion_values[k]
            self._set_k(k)
            self.fit(X)
            value = criterion.value(self)
            criterion_values[k] = value
            return value
        
        low, high, size = min_k, max_k, max_k - min_k
        phi_low, phi_high = round(high - size * PHI), round(low + size * PHI)
        while phi_high >= phi_low + precision:
            if get_value(phi_high) < get_value(phi_low):
                low = phi_low
            else:
                high = phi_high
            size = high - low
            phi_low, phi_high = round(high - size * PHI), round(low + size * PHI)
        get_value(low), get_value(phi_low), get_value(high)
        best_k = min(criterion_values, key=criterion_values.get)
        self._set_k(best_k)
        return best_k
            
