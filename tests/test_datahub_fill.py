import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
os.sys.path.append(os.getcwd() + "/src")

import pandas as pd
import numpy as np

import synthdata as sd
import synthdata.generator as gen

test_1 = pd.read_csv("datasets/test_1.csv")

### Test text values

d = sd.DataHub()
d.load(test_1.copy())

d.data.at[0, 'Species'] = 'BadData'
d.data.at[72, 'Species'] = 'BadData'
d.data.at[149, 'Species'] = 'BadData'

gmm_fill = d.fill(gen.GMM())
kde_fill = d.fill(gen.KDE())

print(f"Expected: {test_1.at[0, 'Species']} - GMM: {gmm_fill.at[0, 'Species']} - KDE: {kde_fill.at[0, 'Species']}")
print(f"Expected: {test_1.at[72, 'Species']} - GMM: {gmm_fill.at[72, 'Species']} - KDE: {kde_fill.at[72, 'Species']}")
print(f"Expected: {test_1.at[149, 'Species']} - GMM: {gmm_fill.at[149, 'Species']} - KDE: {kde_fill.at[149, 'Species']}")

### Test numeric with target

d = sd.DataHub()
d.load(test_1.copy())

bads = 20
labels = d.data.columns
col = np.random.choice(4, bads)
row = np.random.choice(len(test_1), bads)

for c, r in zip(col, row):
    d.data.at[r, labels[c]] = np.nan
    
gmm_fill = d.fill(gen.GMM(), target='Species')
kde_fill = d.fill(gen.KDE(), target='Species')

gmm_error = 0
kde_error = 0
for c, r in zip(col, row):
    gmm_error += abs(test_1.at[r, labels[c]] - gmm_fill.at[r, labels[c]]) ** 2
    kde_error += abs(test_1.at[r, labels[c]] - kde_fill.at[r, labels[c]]) ** 2
    
print()
print("Mean Squared Error of gmm fill", gmm_error / bads)
print("Mean Squared Error of kde fill", kde_error / bads)