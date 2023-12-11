import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import synthdata as sd

# Datasets
path = os.getcwd() + "/datasets"
if path not in os.sys.path:
    os.sys.path.append(path)
from sphere_dataset import sphere
from normal_dataset import normal

interval = (10, 1000)
sections = 20
folds = 20
points = np.array(np.square(np.linspace(*np.sqrt(interval), sections)), dtype=int)
dim = 5
N = 500

dh = sd.DataHub(remove_cov=False)

for generator, name in zip((sphere, normal), ('Esfera', 'Normal')):
    data = generator(interval[1], dim)
    dh.load(data)
    encoders = dh.encoders
    
    mse_GMM = np.zeros(sections)
    mse_KDE = np.zeros(sections)
    
    for i, samples in enumerate(points):
        for fold in range(folds):
            extra = generator(N, dim)
            bad_extra = extra.copy()
            errors = np.random.choice(dim, N)
            for e, error in enumerate(errors):
                bad_extra.iloc[e, error] = np.nan
            dh.load(pd.concat([bad_extra, data[:samples]]), encoders=encoders)
            
            filled = dh.fill(model=sd.generator.GMM())
            mse = 0
            for e, error in enumerate(errors):
                mse += (extra.iloc[e, error] - filled.iloc[i, error]) ** 2
            mse_GMM[i] += mse / N / folds
            
            filled = dh.fill(model=sd.generator.KDE())
            mse = 0
            for e, error in enumerate(errors):
                mse += (extra.iloc[e, error] - filled.iloc[i, error]) ** 2
            mse_KDE[i] += mse / N / folds
    
    plt.figure(figsize=(12, 8))
    plt.plot(points, mse_GMM, color='blue', label='GMM')
    plt.plot(points, mse_KDE, color='red', label='KDE')
    plt.title(f"Error mitja quadrat de la {name.lower()} {dim}-D")
    plt.xlabel("mostres d'entrenament")
    plt.ylabel('MSE')
    plt.ylim(0, 3 * np.mean(mse_KDE))
    plt.legend()
    plt.show()

