import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import synthdata as sd

# Sphere dataset
path = os.getcwd() + "/datasets"
if path not in os.sys.path:
    os.sys.path.append(path)
from sphere_dataset import sphere
from normal_dataset import normal

N = 1000
folds = 10
samples = 500
dh = sd.DataHub()

print(f"\nCorrelated data (3D-sphere)\n{'=' * 50}")
dh.load(sphere(N, 3))
h_modes = 'tune', 'constant', 1e-10
for h_mode in h_modes:
    print(f"Selecting h with {h_mode} option")
    model = sd.generator.KDE(h=h_mode)
    out_llh = dh.kfold_validation(train_samples=samples, folds=folds, model=model)
    out_emd = dh.kfold_validation(train_samples=samples, folds=folds, model=model, validation=sd.validator.EMD)
    found_h = model.h
    print(f"Loglikelihood = {out_llh['validation']}")
    print(f"Earth Movers Distance = {out_emd['validation']}")
    print(f"Execution time (with serach) = {out_llh['fitting_time']}")
    print(f"h = {found_h}")
    
print(f"\nUnorrelated data (3D-sphere)\n{'=' * 50}")
dh.load(normal(N, 3))
h_modes = 'tune', 'constant', 1e-10
for h_mode in h_modes:
    print(f"Selecting h with {h_mode} option")
    model = sd.generator.KDE(h=h_mode)
    out_llh = dh.kfold_validation(train_samples=samples, folds=folds, model=model)
    out_emd = dh.kfold_validation(train_samples=samples, folds=folds, model=model, validation=sd.validator.EMD)
    found_h = model.h
    print(f"Loglikelihood = {out_llh['validation']}")
    print(f"Earth Movers Distance = {out_emd['validation']}")
    print(f"Execution time (with serach) = {out_llh['fitting_time']}")
    print(f"h = {found_h}")