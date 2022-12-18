import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd

import synthdata as sd

iris = pd.read_csv("datasets/iris_dataset.csv")
d = sd.DataHub()
d.load(iris)

X = d.transform(d.data, refit=True)
Y = d.inv_transform(X)
