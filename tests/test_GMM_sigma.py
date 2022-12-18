import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import numpy as np

import synthdata as sd

# Sphere dataset
path = os.getcwd() + "/datasets"
if path not in os.sys.path:
    os.sys.path.append(path)
from sphere_dataset import sphere

dh = sd.DataHub()
small_dh = sd.DataHub()
N = 1000
folds = 5
samples = 500
for dim in [2, 5, 10, 20, 50]:
    print(f"\n{dim}D-sphere\n{'=' * 50}")
    dh.load(sphere(N, dim))
    is_multivariate = True, False
    for multivariate in is_multivariate:
        multivariate_text = 'Multivaraite' if multivariate else 'Univariate'
        print(f"GMM ({multivariate_text} distribution)")
        model = sd.generator.GMM(multivariate=multivariate)
        out = dh.kfold_validation(train_samples=samples, folds=folds, model=model)
        found_k = []
        for fold in range(folds):
            small_dh.load(dh.extend(n_samples=samples))
            small_dh.fill(model=model) # Fit and don't do anything else
            found_k.append(model.k)
        different_k, count = np.unique(found_k, return_counts=True)
        different_k = different_k[count == np.max(count)]
        optimal_k = int(np.round(np.mean(different_k)))
        print(f"Loglikelihood = {out['validation']}")
        print(f"Execution time (with serach) = {out['fitting_time']}")
        print(f"k values: {found_k}, optimal ~ {optimal_k}")
        model._set_k(optimal_k)
        out = dh.kfold_validation(train_samples=500, folds=5, model=model)
        print(f"Execution time (on optimal) = {out['fitting_time']}")
        for k_test in [5, 10, 50]:
            model._set_k(k_test)
            out = dh.kfold_validation(train_samples=500, folds=5, model=model)
            print(f"Execution time (k = {k_test}) = {out['fitting_time']}")