# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model

@author: david
"""

import numpy as np
from time import time as gettime
import matplotlib.pyplot as plt

class GMM:
    def probability_plot(X, k_max):
        dim = X.shape[1]
        n = X.shape[0]
        gmm = GMM()
        probs = []
        ks = []
        for j in range(k_max):
            k = j + 1
            gmm.set_k(k)
            gmm.model(X)
            
            probs.append(
                k * (dim * (dim + 1) / 2 + dim + 1) * np.log(n)
                - 2 * np.log(gmm.likelihood(X))
                )
            ks.append(k)
            
        fig, ax = plt.subplots(1, 1)
        ax.plot(ks, probs)
        plt.show()
    
    def __init__(self, k='auto', k_max=None):
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
        self.inv_cov = self.covariances
        
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
    
    def _inv_cov(self):
        tol = 1e-6
        self.inv_cov = np.array([np.linalg.inv(cov + tol * np.identity(self.dim)) for cov in self.covariances])
    
    def k_probability(self, X=None):
        if X is None:
            X = self.X
        assert X.shape[1] == self.dim, "Size mismatch"
        if self.inv_cov is None:
            self._inv_cov()
        k_prob = []
        tol = 1e-6
        for weight, mean, covariance, inv_cov in \
            zip(self.weights, self.means, self.covariances, self.inv_cov):
                centered = X - mean
                _covariance = covariance + tol * np.identity(self.dim)
                determinant = np.linalg.det(_covariance)
                exponent = -np.sum(centered.dot(inv_cov) * centered, 1) / 2
                k_prob.append(weight * np.exp(exponent) / np.sqrt(determinant))
                    
        k_prob = np.array(k_prob).transpose() / np.power(2 * np.pi, self.dim / 2)
        return np.nan_to_num(k_prob)
        
    
    def probabilities(self, X):
        return np.sum(self.k_probability(X), 1)
    
    def likelihood(self, X):
        return np.prod(self.probabilities(X)) + 1e-300
    
    def iterate(self):
        probs = self.k_probability().transpose()
        resps = probs / np.sum(probs, 0)
        
        t_resps = np.sum(resps, 1)
        self.weights = t_resps / self.n
        self.means = np.transpose(resps.dot(self.X).transpose() / t_resps)
        self.covariances = np.array([
            (self.X - mean).transpose().dot(np.diag(resp).dot(self.X - mean)) / t_resp 
            for t_resp, resp, mean in 
                zip(t_resps, resps, self.means)
                ])
        self.inv_cov = None
        
    def model(self, X, max_iter=1000, max_time=1, llh_tol=1e-3, n_attempts=3):
        self.X = X
        self.n, self.dim = X.shape
        if self.k_mode == 'auto':
            best_bic = None
            best_k = None
            k = 1
            k_lim = 2
            loop = k_lim if self.k_max is None else self.k_max
            while not loop == 0:
                loop -= 1
                gmm = GMM(k)
                gmm.model(X, max_iter, max_time, llh_tol, n_attempts)
                bic = k * (self.dim * (self.dim + 1) / 2 + self.dim + 1) * np.log(self.n) \
                    - 2 * np.log(gmm.likelihood(X))
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
            llh = np.log(self.likelihood(X))
            while loop:
                self.iterate()
                iters -= 1
                last_llh = llh
                llh = np.log(self.likelihood(X))
                loop &= iters > 0
                loop &= llh - last_llh > llh_tol
                loop &= time > gettime()
            if best_llh is None or llh > best_llh:
                best_llh = llh
                best_params = (self.weights, self.means, self.covariances)
        self.weights, self.means, self.covariances = best_params
        self._inv_cov()
            
    def generate(self, size):
        ind, num = np.unique(np.random.choice(self.k, size, p=self.weights), return_counts=True)
        s = [
            np.random.multivariate_normal(self.means[i], self.covariances[i], n)
            for i, n in zip(ind, num)
            ]
        S = np.concatenate(s)
        return S
    
    def fill(self, Y):
        assert Y.shape[1] == self.dim, "Size mismatch"
        if self.inv_cov is None:
            self._inv_cov()
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

if __name__ == '__main__':
    n = 100
    
    phase = np.linspace(0, 2 * np.pi, n)
    noise = [np.random.normal(size=n) for _ in range(2)]
    data = np.array([
            np.sin(phase + noise[0]) + .1 * np.sin(phase + noise[1]),
            np.cos(phase + noise[0]) + .1 * np.cos(phase + noise[1])
        ]).transpose() * 10 + np.array([1, 5])
    
    mean = np.mean(data, 0)
    var = np.var(data, 0)
    
    X = (data - mean) / np.sqrt(var)
    
    gmm = GMM()
    gmm.model(X)
    
    syn = 1000
    fdata = np.random.normal(mean, np.sqrt(var) / 2, size=(syn, 2)) + np.full((syn, 2), [np.nan, 0])
    F = gmm.fill((fdata - mean) / np.sqrt(var))
    S = gmm.generate(syn)
    
    sdata = S * np.sqrt(var) + mean
    fdata = F * np.sqrt(var) + mean
    
    fig, ax = plt.subplots(1, 1)
    sp = ax.scatter(sdata[:,0], sdata[:,1], c='yellow', alpha=.2)
    sp = ax.scatter(fdata[:,0], fdata[:,1], c='green', alpha=.1)
    sp = ax.scatter(data[:,0], data[:,1], c='red', alpha=.2)

    plt.show()
    
    #GMM.probability_plot(X, 10)
