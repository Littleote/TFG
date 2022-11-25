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

def ideal_FUN(trans, subdata, n_samples, folds):
    X = trans.transform(subdata, process=False)
    ind = np.random.choice(len(subdata), (folds, n_samples), False)
    emd = 0
    for fold in range(folds):
        emd += val.EMD(X[ind[fold]], X[ind[(fold + 1) % folds]])
    return emd / folds

def main():
    test_2 = pd.read_csv("datasets/ideal-all.csv", sep=';')
    d = sd.DataHub(cores=3)
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
    interval = (5, 500)
    n_samples = np.round(np.square(np.linspace(np.sqrt(interval[0]), np.sqrt(interval[1]), sections))).astype(int)
    ideal_emd = {str(ideal): np.zeros(sections) for ideal in ideals}
    
    methods = {
        'gmm': gen.GMM(),
        'kde': gen.KDE(),
        'vae': gen.VAE(),
        'vae_cpu': gen.VAE(device='cpu')
    }
    names = methods.keys()
    colors = ('blue', 'orange', 'green', 'red')
    
    kfold_args = {
        'folds': 7,
        'return_fit': True,
        'return_time': True,
        'target': 'ideal',
        'validation': val.EMD
    }
    
    time_fit = {name: np.zeros(sections) for name in names}
    time_eval = {name: np.zeros(sections) for name in names}
    emd_train = {name: {str(ideal): np.zeros(sections) for ideal in ideals} for name in names}
    emd_val = {name: {str(ideal): np.zeros(sections) for ideal in ideals} for name in names}
    
    sizes = d.for_target('ideal', lambda trans, x: len(x))
    for ideal in ideals:
        ideal = str(ideal)
        print(f"Ideal value {ideal} with {sizes[ideal]} samples total")
    
    start = gettime()
    print("Tunning VAE hyperparameters ... ", end='')
    present = d.transform(d.extend(interval[1], target='ideal'), refit=True)
    methods['vae'].prefit(present)
    methods['vae_cpu'].prefit(present)
    print(f"in {gettime() - start} seconds")
    
    for i in range(sections):
        print(f"Progress: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
        for name in names:
            start = gettime()
            print(f"Running {name.upper()} for n_samples = {n_samples[i]} ... ", end='')
            emd_results = d.kfold_validation(n_samples=n_samples[i], **kfold_args, method=methods[name])
            print(f"in {np.round(gettime() - start, 4)} seconds")
            for ideal in ideals:
                ideal = str(ideal)
                emd_val[name][ideal][i] = emd_results[ideal]['validation']
                emd_train[name][ideal][i] = emd_results[ideal]['train']
                time_fit[name][i] += emd_results[ideal]['fitting_time']
                time_eval[name][i] += emd_results[ideal]['evaluation_time']
            
        _ideal_emd = d.for_target(target='ideal', FUN=ideal_FUN, n_samples=n_samples[i], folds=15)
        for ideal in ideals:
            ideal = str(ideal)
            ideal_emd[ideal][i] = _ideal_emd[ideal]
    
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
        plt.plot(n_samples, ideal_emd[ideal], label="Ideal", color='black')
        for i, name in enumerate(names):
            plt.plot(n_samples, emd_val[name][ideal], label=f"{name.upper()} (Validation)", color=colors[i])
            plt.plot(n_samples, emd_train[name][ideal], label=f"{name.upper()} (Train)", color=colors[i], dashes=[1, 2, 5, 2])
        plt.title("Ideal: " + ideal)
        plt.yscale('log')
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()