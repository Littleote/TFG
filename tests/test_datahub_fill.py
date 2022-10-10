import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
os.sys.path.append(os.getcwd() + "/src")

import pandas as pd

import synthdata as sd
import synthdata.generator as gen

test_1 = pd.read_csv("datasets/test_1.csv")
d = sd.DataHub()
d.load(test_1.copy())

d.data.at[0, 'Species'] = 'BadData'
d.data.at[72, 'Species'] = 'BadData'
d.data.at[149, 'Species'] = 'BadData'

gmm_fill = d.fill(gen.GMM(), preprocess='normalize')
kde_fill = d.fill(gen.KDE(), preprocess='whitening')

print(f"Expected: {test_1.at[0, 'Species']} - GMM: {gmm_fill.at[0, 'Species']} - KDE: {kde_fill.at[0, 'Species']}")
print(f"Expected: {test_1.at[72, 'Species']} - GMM: {gmm_fill.at[72, 'Species']} - KDE: {kde_fill.at[72, 'Species']}")
print(f"Expected: {test_1.at[149, 'Species']} - GMM: {gmm_fill.at[149, 'Species']} - KDE: {kde_fill.at[149, 'Species']}")
    
  