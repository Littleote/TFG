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
from synthdata.encoder import ignore

def main():
    # Datasets
    path = os.getcwd() + "/datasets"
    if path not in os.sys.path:
        os.sys.path.append(path)
    ideal_reduced = pd.read_csv("datasets/ideal_reduced_dataset.csv", sep=';')

    d = sd.DataHub(cores=2)
    d.load(ideal_reduced, encoders = {
        'time_x': ignore(0),
        'time_y': ignore(0),
        'time_x.1': ignore(0),
        'time_y.1': ignore(0),
        'time_log': ignore(0),
        'id': ignore(0),
        'threads': ignore(0),
        'P_index': ignore(),
        'T_list': ignore(),
        'P_list': ignore()
        })
    
    ideals = np.sort(d.data['ideal'].unique())
    sections = 20
    interval = (5, 500)
    samples = np.round(np.square(np.linspace(np.sqrt(interval[0]), np.sqrt(interval[1]), sections))).astype(int)
    
    models = {
        'gmm': gen.GMM(),
        'kde': gen.KDE(),
    }
    names = models.keys()
    colors = ('blue', 'red')
    
    kfold_args = {
        'folds': 10,
        'validation_samples': 100,
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
        print(f"Progres: {i} / {sections}, {np.round(100 * i / sections, 2)}%")
        for name in names:
            llh_results = d.kfold_validation(train_samples=samples[i], **kfold_args, model=models[name])
            for ideal in ideals:
                ideal = str(ideal)
                llh_val[name][ideal][i] = llh_results[ideal]['validation']
                llh_train[name][ideal][i] = llh_results[ideal]['train']
                time_fit[name][i] += llh_results[ideal]['fitting_time']
                time_eval[name][i] += llh_results[ideal]['evaluation_time']
    
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(names):
        plt.plot(samples, time_fit[name], label=name.upper(), color=colors[i])
    plt.title("Temps d'entrenament (segons)")
    plt.xlabel("Mostres d'entrenament")
    plt.ylabel('Temps')
    plt.legend()
    plt.show()
    
    for ideal in ideals:
        ideal = str(ideal)
        plt.figure(figsize=(12, 8))
        for i, name in enumerate(names):
            plt.plot(samples, llh_val[name][ideal], label=f"{name.upper()} (Validation)", color=colors[i])
            plt.plot(samples, llh_train[name][ideal], label=f"{name.upper()} (Train)", color=colors[i], dashes=[1, 2, 5, 2])
        plt.title(f"Logversemblança pel valor ideal numero {ideal}")
        plt.xlabel("Mostres d'entrenament")
        plt.ylabel('Logversemblança')
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()