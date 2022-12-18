import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotting(gen, name=None, x1=0, x2=1, y=2):
    X1 = gen.to_numpy()[:, x1]
    X2 = gen.to_numpy()[:, x2]
    Y = gen.to_numpy()[:, y]
    _, Y = np.unique(Y, return_inverse=True)

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X1, X2, c=Y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel(gen.columns[0])
    plt.ylabel(gen.columns[1])
    plt.title(name)
    
    plt.show()

import synthdata as sd
import synthdata.generator as gen

def main():
    iris = pd.read_csv("datasets/iris_dataset.csv")
    d = sd.DataHub(cores=2)
    d.load(iris)
    
    plotting(iris, 'Original', y=4)
    plotting(d.generate(100, target='Species', model=gen.GMM()), 'Gaussian Mixture Model', y=4)
    plotting(d.generate(100, target='Species', model=gen.KDE()), 'Kernel Density Estimator', y=4)
    
if __name__ == "__main__":
    main()