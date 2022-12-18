import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)
    
import pandas as pd
import numpy as np

import synthdata as sd
import synthdata.generator as gen

iris = pd.read_csv("datasets/iris_dataset.csv")

### Test text values

d = sd.DataHub()
d.load(iris.copy())

d.data.at[0, 'Species'] = 'BadData'
d.data.at[72, 'Species'] = 'BadData'
d.data.at[149, 'Species'] = 'BadData'

gmm_fill = d.fill(gen.GMM())
kde_fill = d.fill(gen.KDE())

print(f"Expected: {iris.at[0, 'Species']} - GMM: {gmm_fill.at[0, 'Species']} - KDE: {kde_fill.at[0, 'Species']}")
print(f"Expected: {iris.at[72, 'Species']} - GMM: {gmm_fill.at[72, 'Species']} - KDE: {kde_fill.at[72, 'Species']}")
print(f"Expected: {iris.at[149, 'Species']} - GMM: {gmm_fill.at[149, 'Species']} - KDE: {kde_fill.at[149, 'Species']}")

### Test numeric with target

d = sd.DataHub()
d.load(iris.copy())

bads = 20
labels = d.data.columns
col = np.random.choice(4, bads)
row = np.random.choice(len(iris), bads)

for c, r in zip(col, row):
    d.data.at[r, labels[c]] = np.nan
    
gmm_fill = d.fill(gen.GMM(), target='Species')
kde_fill = d.fill(gen.KDE(), target='Species')

gmm_error = 0
kde_error = 0
for c, r in zip(col, row):
    gmm_error += abs(iris.at[r, labels[c]] - gmm_fill.at[r, labels[c]]) ** 2
    kde_error += abs(iris.at[r, labels[c]] - kde_fill.at[r, labels[c]]) ** 2
    
print()
print("Mean Squared Error of gmm fill", gmm_error / bads)
print("Mean Squared Error of kde fill", kde_error / bads)