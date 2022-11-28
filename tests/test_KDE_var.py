import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import synthdata as sd
import synthdata.generator as gen
import synthdata.validator as val
from synthdata.encoder import ignore

def main():
    test_2 = pd.read_csv("datasets/ideal-all.csv", sep=';')
    d = sd.DataHub(remove_cov=False)
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
    sections = 50
    kfold_args = {
        'n_samples': 100,
        'folds': 5,
        'validation': val.EMD,
        'return_fit': True,
        'return_time': False,
        'target': 'ideal'
    }
    var = np.square(np.linspace(np.sqrt(1e-6), np.sqrt(1), sections))
    kde_llh = {str(ideal): np.zeros(sections) for ideal in ideals}
    kde_llh_ = {str(ideal): np.zeros(sections) for ideal in ideals}
    
    sizes = d.for_target('ideal', lambda trans, x: len(x))
    for ideal in ideals:
        ideal = str(ideal)
        print(f"Ideal value {ideal} with {sizes[ideal]} samples total")
    
    for i in range(sections):
        print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
        _kde_llh = d.kfold_validation(**kfold_args, method=gen.KDE(var=var[i]))
        for ideal in ideals:
            ideal = str(ideal)
            kde_llh[ideal][i] = _kde_llh[ideal]['validation']
            kde_llh_[ideal][i] = _kde_llh[ideal]['train']
    
    for ideal in ideals:
        ideal = str(ideal)
        plt.plot(var, kde_llh[ideal], label="KDE (Validation)", color='orange')
        plt.plot(var, kde_llh_[ideal], label="KDE (Train)", color='orange', dashes=[1, 2, 5, 2])
        plt.title("Ideal: " + ideal)
        plt.xscale("log")
        plt.ylim([0.9 * np.min(kde_llh_[ideal]), 1.1 * np.max(kde_llh_[ideal])])
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()