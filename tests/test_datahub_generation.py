import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
os.sys.path.append(os.getcwd() + "/src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotting(gen, name=None):
    X = gen.to_numpy()[:, :2]
    y = gen.to_numpy()[:, 4]
    _, y = np.unique(y, return_inverse=True)

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel(gen.columns[0])
    plt.ylabel(gen.columns[1])
    plt.title(name)
    
    plt.show()

import synthdata as sd
import synthdata.generator as gen

test_1 = pd.read_csv("datasets/test_1.csv")
d = sd.DataHub()
d.load(test_1)

plotting(test_1, 'Original')
plotting(d.generate(100, target='Species', method=gen.GMM()), 'Gaussian Mixture Model')
plotting(d.generate(100, target='Species', method=gen.KDE()), 'Kernel Density Estimator')
 