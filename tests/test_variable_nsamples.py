import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time as gettime

import synthdata as sd
import synthdata.generator as gen
from synthdata.encoder import ignore
# import synthdata.validator as val

test_2 = pd.read_csv("datasets/ideal-all.csv", sep=';')
d = sd.DataHub()
d.load(test_2, encoders = {
    'time_x': ignore(0),
    'time_y': ignore(0),
    'time_x.1': ignore(0),
    'time_y.1': ignore(0),
    'time_log': ignore(0),
    'id': ignore(0),
    'threads': ignore(0),
    'P_index': ignore(0),
    'T_list': ignore(),
    'P_list': ignore()
    })

ideals = np.sort(test_2['ideal'].unique())
sections = 10
n_samples = np.round(np.exp(np.linspace(np.log(5), np.log(500), sections))).astype(int)
gmm_llh = {str(ideal): np.zeros(sections) for ideal in ideals}
kde_llh = {str(ideal): np.zeros(sections) for ideal in ideals}

for i in range(sections):
    print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
    start = gettime()
    print(f"Running GMM for n_seamples = {n_samples[i]} ... ", end='')
    _gmm_llh = d.kfold_validation(n_samples=n_samples[i], folds=5, target='ideal', method=gen.GMM())
    print(f"in {gettime() - start} seconds")
    start = gettime()
    print(f"Running KDE for n_seamples = {n_samples[i]} ... ", end='')
    _kde_llh = d.kfold_validation(n_samples=n_samples[i], folds=5, target='ideal', method=gen.KDE())
    print(f"in {gettime() - start} seconds")
    for ideal in ideals:
        ideal = str(ideal)
        gmm_llh[ideal][i] = _gmm_llh[ideal]
        kde_llh[ideal][i] = _kde_llh[ideal]

for ideal in ideals:
    ideal = str(ideal)
    plt.plot(n_samples, gmm_llh[ideal])
    plt.plot(n_samples, kde_llh[ideal])
    plt.title("Ideal: " + ideal)
    plt.show()