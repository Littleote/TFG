import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd

import synthdata as sd
import synthdata.generator as gen
from synthdata.encoder import ignore
from time import time as gettime

def main():
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
    
    device = 'cuda:0'
    model = gen.VAE(device=device, fit_layers=5, fit_enc_dim=15)
    subset = d.extend(10000, 20000, target='ideal')
    start = gettime()
    d.run(subset, model.fit)
    delta = gettime() - start
    print(f"Executed VAE in {device} for {len(subset)} samples in {round(delta, 4)}s.")
        
if __name__ == "__main__":
    main()