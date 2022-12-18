import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd

import synthdata as sd
from synthdata.validator import EMD

iris = pd.read_csv("datasets/iris_dataset.csv")
d = sd.DataHub()
d.load(iris)


gmm = d.transform(d.generate(len(iris), model=sd.generator.GMM()))
kde = d.transform(d.generate(len(iris), model=sd.generator.KDE()))
vae = d.transform(d.generate(len(iris), model=sd.generator.VAE()))
original = d.transform(d.data)

print("GMM:", EMD(original, gmm))
print("KDE:", EMD(original, kde))
print("VAE:", EMD(original, vae))

sd.encoder.greater(2)
