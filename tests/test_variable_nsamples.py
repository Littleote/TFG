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

def main():
    test_2 = pd.read_csv("datasets/ideal-all.csv", sep=';')
    d = sd.DataHub(cores=4)
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
    gmm_time = np.zeros(sections)
    kde_time = np.zeros(sections)
    gmm_llh = {str(ideal): np.zeros(sections) for ideal in ideals}
    gmm_llh_ = {str(ideal): np.zeros(sections) for ideal in ideals}
    kde_llh = {str(ideal): np.zeros(sections) for ideal in ideals}
    kde_llh_ = {str(ideal): np.zeros(sections) for ideal in ideals}
    
    sizes = d.for_target('ideal', lambda trans, x: len(x))
    for ideal in ideals:
        ideal = str(ideal)
        print(f"Ideal value {ideal} with {sizes[ideal]} samples total")
    
    for i in range(sections):
        print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
        start = gettime()
        print(f"Running GMM for n_seamples = {n_samples[i]} ... ", end='')
        _gmm_llh = d.kfold_validation(n_samples=n_samples[i], folds=5, target='ideal', return_fit=True, method=gen.GMM())
        gmm_time[i] = gettime() - start
        print(f"in {gmm_time[i]} seconds")
        start = gettime()
        print(f"Running KDE for n_seamples = {n_samples[i]} ... ", end='')
        _kde_llh = d.kfold_validation(n_samples=n_samples[i], folds=5, target='ideal', return_fit=True, method=gen.KDE())
        kde_time[i] = gettime() - start
        print(f"in {kde_time[i]} seconds")
        for ideal in ideals:
            ideal = str(ideal)
            gmm_llh[ideal][i] = _gmm_llh[ideal]['validation']
            gmm_llh_[ideal][i] = _gmm_llh[ideal]['train']
            kde_llh[ideal][i] = _kde_llh[ideal]['validation']
            kde_llh_[ideal][i] = _kde_llh[ideal]['train']
    
    plt.plot(n_samples, gmm_time, label="GMM", color='blue')
    plt.plot(n_samples, kde_time, label="KDE", color='orange')
    plt.title("Execution time")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    for ideal in ideals:
        ideal = str(ideal)
        plt.plot(n_samples, gmm_llh[ideal], label="GMM (Validation)", color='blue')
        plt.plot(n_samples, gmm_llh_[ideal], label="GMM (Train)", color='blue', dashes=[1, 2, 5, 2])
        plt.plot(n_samples, kde_llh[ideal], label="KDE (Validation)", color='orange')
        plt.plot(n_samples, kde_llh_[ideal], label="KDE (Train)", color='orange', dashes=[1, 2, 5, 2])
        plt.title("Ideal: " + ideal)
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()