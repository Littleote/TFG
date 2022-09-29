# -*- coding: utf-8 -*-
"""
Data Class

@author: david
"""

import numpy as np
import pandas as pd

import Encoder
import GMM
import KDE
#import VAE

class Data:
    def __init__(self):
        pass
    
    def load(self, data, **options):
        assert len(data.shape) == 2, "data must be a 2d array of shape (n_samples, n_features)"
        
        self.data = data
        self.labels = list(data.columns)
        self.dtypes = data.dtypes.to_dict()
        self.encoders = {
            label: Encoder.auto(data[label])
            for label in self.labels
            } | options.get('encoders', dict())
        self.samples, self.features = data.shape
    
    def set_encoder(self, label, encoder):
        self.encoders[label] = encoder
        
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
        return X
    
    def inv_transform(self, X):
        Xfeatures = sum([enc.size for enc in self.encoders.values()])
        assert X.shape[1] == Xfeatures, f"X n_features ({X.shape[1]}) different from expected ({sum([enc.size for enc in self.encoders.values()])})"
        data = pd.DataFrame(columns=self.labels, dtype=object)
        i = 0
        for label in self.labels:
            size = self.encoders[label].size
            data[[label]] = self.encoders[label].decode(X[:,i:i + size])
            i += size
        return data.astype(self.dtypes)
    
    def generate(self, n_samples, method, target=None, **model_args):
        if target is None:
            X = self.transform()
            nans = np.isnan(X).any(axis=1)
            print(nans)
            if nans.all():
                raise ValueError("All rows contain nan")
            X_ = X[np.logical_not(nans)]
            method.model(X_, **model_args)
            return self.inv_transform(method.generate(n_samples))
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = Encoder.EncoderIgnore(default=value)
                X = self.transform(self.data[self.data[target] == value])
                nans = np.isnan(X).any(axis=1)
                if nans.all():
                    raise ValueError("All rows for {target} = {value} contain nan")
                X_ = X[np.logical_not(nans)]
                method.model(X_, **model_args)
                new_data.append(self.inv_transform(method.generate(n_samples)))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
            
    def fill(self, method, target=None, **model_args):
        if target is None:
            X = self.transform()
            nans = np.isnan(X).any(axis=1)
            if nans.all():
                raise ValueError("All rows contain nan")
            method.model(X[np.logical_not(nans)], **model_args)
            X[nans] = method.fill(X[nans])
            return self.inv_transform(X)
        else:
            target_encoder = self.encoders[target]
            new_data = []
            for value in self.data[target].unique():
                self.encoders[target] = Encoder.EncoderIgnore(default=value)
                X = self.transform(self.data[self.data[target] == value])
                nans = np.isnan(X).any(axis=1)
                if nans.all():
                    raise ValueError("All rows for {target} = {value} contain nan")
                method.model(X[np.logical_not(nans)], **model_args)
                X[nans] = method.fill(X[nans])
                new_data.append(self.inv_transform(X))
            self.encoders[target] = target_encoder
            return pd.concat(new_data)
        
    

if __name__ == "__main__":
    test_1 = pd.read_csv("test_1.csv")
    d = Data()
    d.load(test_1.copy())
    X = d.transform()
    Y = d.inv_transform(X)
    
    import matplotlib.pyplot as plt
    def plotting(gen):
        X = gen.to_numpy()[:, :2]
        y = gen.to_numpy()[:, 4]
        _, y = np.unique(y, return_inverse=True)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        
        plt.show()
    
    plotting(test_1)
    plotting(d.generate(100, target='Species', method=GMM.GMM()))
    plotting(d.generate(100, target='Species', method=KDE.KDE()))
    
    
    d.data.at[0, 'Species'] = 'BadData'
    d.data.at[72, 'Species'] = 'BadData'
    d.data.at[149, 'Species'] = 'BadData'
    gmm_fill = d.fill(GMM.GMM())
    kde_fill = d.fill(KDE.KDE())
    print(f"Expected: {test_1.at[0, 'Species']} - GMM: {gmm_fill.at[0, 'Species']} - KDE: {kde_fill.at[0, 'Species']}")
    print(f"Expected: {test_1.at[72, 'Species']} - GMM: {gmm_fill.at[72, 'Species']} - KDE: {kde_fill.at[72, 'Species']}")
    print(f"Expected: {test_1.at[149, 'Species']} - GMM: {gmm_fill.at[149, 'Species']} - KDE: {kde_fill.at[149, 'Species']}")
    
    