import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import synthdata as sd

# Sphere dataset
N = 1000
dim = 3
path = os.getcwd() + "/datasets"
if path not in os.sys.path:
    os.sys.path.append(path)
from sphere_dataset import sphere

dh = sd.DataHub()
dh.load(sphere(N, dim))
criterions = 'Bayesian', 'Akaike', 'Cross-Validation'
for criterion in criterions:
    print(f"Criterion: {criterion}")
    model = sd.generator.GMM(criterion=criterion)
    out = dh.kfold_validation(train_samples=500, folds=5, model=model)
    print(f"Loglikelihood = {out['validation']}")
    print(f"Execution time = {out['fitting_time']}")
    print(f"k = {model.k}")
