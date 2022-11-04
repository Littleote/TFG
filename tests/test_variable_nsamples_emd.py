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
import synthdata.validator as val

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
n_samples = np.round(np.square(np.linspace(np.sqrt(5), np.sqrt(500), sections))).astype(int)
n_samples = [500]
ideal_emd = {str(ideal): np.zeros(sections) for ideal in ideals}

gmm_time_fit = np.zeros(sections)
gmm_time_eval = np.zeros(sections)
gmm_emd_train = {str(ideal): np.zeros(sections) for ideal in ideals}
gmm_emd_val = {str(ideal): np.zeros(sections) for ideal in ideals}

kde_time_fit = np.zeros(sections)
kde_time_eval = np.zeros(sections)
kde_emd_train = {str(ideal): np.zeros(sections) for ideal in ideals}
kde_emd_val = {str(ideal): np.zeros(sections) for ideal in ideals}

vae_time_fit = np.zeros(sections)
vae_time_eval = np.zeros(sections)
vae_emd_train = {str(ideal): np.zeros(sections) for ideal in ideals}
vae_emd_val = {str(ideal): np.zeros(sections) for ideal in ideals}

sizes = d.for_target('ideal', lambda x: len(x))
for ideal in ideals:
    ideal = str(ideal)
    print(f"Ideal value {ideal} with {sizes[ideal]} samples total")

kfold_args = {
    'folds': 5,
    'return_fit': True,
    'return_time': True,
    'target': 'ideal',
    'validation': val.EMD
}

for i in range(sections):
    print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
    start = gettime()
    print(f"Running VAE for n_seamples = {n_samples[i]} ... ", end='')
    _vae_emd = d.kfold_validation(n_samples=n_samples[i], **kfold_args, method=gen.VAE())
    print(f"in {gettime() - start} seconds")
    start = gettime()
    print(f"Running GMM for n_seamples = {n_samples[i]} ... ", end='')
    _gmm_emd = d.kfold_validation(n_samples=n_samples[i], **kfold_args, method=gen.GMM())
    print(f"in {gettime() - start} seconds")
    start = gettime()
    print(f"Running KDE for n_seamples = {n_samples[i]} ... ", end='')
    _kde_emd = d.kfold_validation(n_samples=n_samples[i], **kfold_args, method=gen.KDE())
    print(f"in {gettime() - start} seconds")
    _ideal_emd = d.kfold_validation(n_samples=n_samples[i], **kfold_args | {'return_fit': False}, method=gen.KDE(var=lambda n, d: 0))
    for ideal in ideals:
        ideal = str(ideal)
        vae_emd_val[ideal][i] = _vae_emd[ideal]['validation']
        vae_emd_train[ideal][i] = _vae_emd[ideal]['train']
        vae_time_fit[i] += _vae_emd[ideal]['fitting']
        vae_time_eval[i] += _vae_emd[ideal]['evaluation']
        gmm_emd_val[ideal][i] = _gmm_emd[ideal]['validation']
        gmm_emd_train[ideal][i] = _gmm_emd[ideal]['train']
        gmm_time_fit[i] += _gmm_emd[ideal]['fitting']
        gmm_time_eval[i] += _gmm_emd[ideal]['evaluation']
        kde_emd_val[ideal][i] = _kde_emd[ideal]['validation']
        kde_emd_train[ideal][i] = _kde_emd[ideal]['train']
        kde_time_fit[i] += _kde_emd[ideal]['fitting']
        kde_time_eval[i] += _kde_emd[ideal]['evaluation']
        ideal_emd[ideal][i] = _ideal_emd[ideal]['evaluation']

plt.plot(n_samples, gmm_time_fit, label="GMM", color='blue')
plt.plot(n_samples, kde_time_fit, label="KDE", color='orange')
plt.plot(n_samples, vae_time_fit, label="VAE", color='green')
plt.title("Fitting time")
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(n_samples, gmm_time_eval, label="GMM", color='blue')
plt.plot(n_samples, kde_time_eval, label="KDE", color='orange')
plt.plot(n_samples, vae_time_eval, label="VAE", color='green')
plt.title("Evaulation time")
plt.yscale('log')
plt.legend()
plt.show()

for ideal in ideals:
    ideal = str(ideal)
    plt.plot(n_samples, ideal_emd[ideal], label="Ideal", color='black')
    plt.plot(n_samples, gmm_emd_val[ideal], label="GMM (Validation)", color='blue')
    plt.plot(n_samples, gmm_emd_train[ideal], label="GMM (Train)", color='blue', dashes=[1, 2, 5, 2])
    plt.plot(n_samples, kde_emd_val[ideal], label="KDE (Validation)", color='orange')
    plt.plot(n_samples, kde_emd_train[ideal], label="KDE (Train)", color='orange', dashes=[1, 2, 5, 2])
    plt.plot(n_samples, vae_emd_val[ideal], label="VAE (Validation)", color='green')
    plt.plot(n_samples, vae_emd_train[ideal], label="VAE (Train)", color='green', dashes=[1, 2, 5, 2])
    plt.title("Ideal: " + ideal)
    plt.yscale('log')
    plt.legend()
    plt.show()