import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import pandas as pd
import numpy as np

import synthdata as sd

# Datasets
path = os.getcwd() + "/datasets"
if path not in os.sys.path:
    os.sys.path.append(path)
iris = pd.read_csv("datasets/iris_dataset.csv")
def analize_iris(train, test):
    from sklearn import pipeline
    from sklearn import preprocessing
    from sklearn import svm
    from sklearn import metrics
    
    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVC(gamma='auto'))
    pipe.fit(train.to_numpy()[:,:4].astype(float), train.to_numpy()[:,4])
    real = test.to_numpy()[:,4]
    pred = pipe.predict(test.to_numpy()[:,:4].astype(float))
    return metrics.balanced_accuracy_score(real, pred)

dh = sd.DataHub()
dh.load(iris)
repeats = 1000

def original_data_value():
    n, half = len(dh.data), int(len(dh.data) / 2)
    ind = np.random.choice(n, n, False)
    return analize_iris(iris.iloc[ind[:half]], iris.iloc[ind[half:]])

def augmented_data_value(model, augment):
    n, half = len(dh.data), int(len(dh.data) / 2)
    ind = np.random.choice(n, n, False)
    dh_sub = sd.DataHub()
    dh_sub.load(iris.iloc[ind[:half]], encoders=dh.encoders)
    return analize_iris(dh_sub.generate(half * augment), iris.iloc[ind[half:]])

results = dict()

results['original'] = [original_data_value() for _ in range(repeats)]

results['KDE'] = [augmented_data_value(sd.generator.KDE(), 1) for _ in range(repeats)]
results['GMM'] = [augmented_data_value(sd.generator.GMM(), 1) for _ in range(repeats)]

results['inc_KDE'] = [augmented_data_value(sd.generator.KDE(), 10) for _ in range(repeats)]
results['inc_GMM'] = [augmented_data_value(sd.generator.GMM(), 10) for _ in range(repeats)]
results['inc_VAE'] = [augmented_data_value(sd.generator.VAE(), 10) for _ in range(repeats)]

output = pd.DataFrame(results)
output.to_csv('ignore/iris_results.csv', sep=';')