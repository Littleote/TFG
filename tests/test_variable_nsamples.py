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
    ideal_reduced = pd.read_csv("datasets/ideal_reduced_dataset.csv", sep=';')
    d = sd.DataHub(cores=4)
    d.load(ideal_reduced, encoders = {
        'time_x': ignore(),
        'time_y': ignore(),
        'time_x.1': ignore(),
        'time_y.1': ignore(),
        'time_log': ignore(),
        'id': ignore(),
        'threads': ignore(),
        'P_index': ignore(),
        'T_list': ignore(),
        'P_list': ignore()
        })
    
    ideals = np.sort(d.data['ideal'].unique())
    sections = 10
    interval = (5, 500)
    n_samples = np.round(np.square(np.linspace(np.sqrt(interval[0]), np.sqrt(interval[1]), sections))).astype(int)
    
    models = {
        'gmm (Multivariate)': gen.GMM(multivariate=True),
        'kde (Tuned)': gen.KDE(h='tune')
    }
    names = models.keys()
    colors = ('blue', 'red')
    
    kfold_args = {
        'folds': 7,
        'return_fit': True,
        'return_time': True,
        'target': 'ideal',
    }
    
    time_fit = {name: np.zeros(sections) for name in names}
    time_eval = {name: np.zeros(sections) for name in names}
    llh_train = {name: {str(ideal): np.zeros(sections) for ideal in ideals} for name in names}
    llh_val = {name: {str(ideal): np.zeros(sections) for ideal in ideals} for name in names}
    
    sizes = d.for_target('ideal', lambda trans, x: len(x))
    for ideal in ideals:
        ideal = str(ideal)
        print(f"Ideal value {ideal} with {sizes[ideal]} samples total")
    
    for i in range(sections):
        print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
        for name in names:
            start = gettime()
            print(f"Running {name.upper()} for n_samples = {n_samples[i]} ... ", end='')
            emd_results = d.kfold_validation(train_samples=n_samples[i], **kfold_args, model=models[name])
            print(f"in {np.round(gettime() - start, 4)} seconds")
            for ideal in ideals:
                ideal = str(ideal)
                llh_val[name][ideal][i] = emd_results[ideal]['validation']
                llh_train[name][ideal][i] = emd_results[ideal]['train']
                time_fit[name][i] += emd_results[ideal]['fitting_time']
                time_eval[name][i] += emd_results[ideal]['evaluation_time']
    
    for i, name in enumerate(names):
        plt.plot(n_samples, time_fit[name], label=name.upper(), color=colors[i])
    plt.title("Fitting time")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    for i, name in enumerate(names):
        plt.plot(n_samples, time_eval[name], label=name.upper(), color=colors[i])
    plt.title("Evaulation time")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    for ideal in ideals:
        ideal = str(ideal)
        for i, name in enumerate(names):
            plt.plot(n_samples, llh_val[name][ideal], label=f"{name.upper()} (Validation)", color=colors[i])
            plt.plot(n_samples, llh_train[name][ideal], label=f"{name.upper()} (Train)", color=colors[i], dashes=[1, 2, 5, 2])
        plt.title("Ideal: " + ideal)
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()