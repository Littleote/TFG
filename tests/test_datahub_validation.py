import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd

import synthdata as sd
import synthdata.generator as gen
import synthdata.validator as val

iris = pd.read_csv("datasets/iris_dataset.csv")
d = sd.DataHub()
d.load(iris)

gmm_llh = d.kfold_validation(folds=3, model=gen.GMM(), return_fit=True)
kde_llh = d.kfold_validation(folds=3, model=gen.KDE(), return_fit=True)

gmm_emd = d.kfold_validation(folds=3, model=gen.GMM(), validation=val.EMD)
kde_emd = d.kfold_validation(folds=3, model=gen.KDE(), validation=val.EMD)

print("Mean loglikelihood of GMM: ", gmm_llh['validation'], " (Mean train llh: ", gmm_llh['train'], ")", sep="")
print("Mean loglikelihood of KDE: ", kde_llh['validation'], " (Mean train llh: ", kde_llh['train'], ")", sep="")

print("Mean EMD of GMM:", gmm_emd['validation'])
print("Mean EMD of KDE:", kde_emd['validation'])
