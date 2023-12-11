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
ideal_reduced = pd.read_csv("datasets/ideal_reduced_dataset.csv", sep = ';')

ignored = [
    'time_x', 'time_y', 'time_x.1', 'time_y.1', 'time_log',
    'id', 'threads', 'P_index', 'T_list', 'P_list',
]
target = 'ideal'
labels = (1 + np.arange(12))
non_target = [name for name in ideal_reduced.columns if name != target]

ordinal = ['affinity', 'comp opt']
ohe = ['label']
numeric = [name for name in non_target if name not in ignored + ordinal + ohe]

def analize_ideal(train, test, labels, **args):
    from sklearn import pipeline
    from sklearn import preprocessing
    from sklearn import compose
    from sklearn import ensemble
    from sklearn import metrics
    
    steps = [
        ('ordinal', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal),
        ('ohe', preprocessing.OneHotEncoder(handle_unknown='ignore'), ohe),
        ('numeric', preprocessing.StandardScaler(), numeric),
    ]
    pipe = pipeline.make_pipeline(
        compose.ColumnTransformer(steps, remainder='drop'),
        preprocessing.StandardScaler(),
        ensemble.RandomForestClassifier(**args)
    )
    pipe.fit(train.loc[:, non_target], train.loc[:,target])
    real = test.loc[:,target].to_numpy()
    preds = pipe.predict(test.loc[:, non_target])
    return metrics.precision_score(real, preds, labels=labels, average=None)

encoders = {name: sd.encoder.ignore() for name in ignored}
repeats = 50

def run_model(model, samples='max_target'):
    def func(df):
        dh_ = sd.DataHub(model=model)
        dh_.load(df, encoders=encoders)
        return (dh_.extend(n_samples=samples, target=target))
    return func

treatments = {
    'Res': lambda df: (df),
    'Reduir': run_model(None, 'min_target'),
    'Clasificador': lambda df: (df),
    'KDE': run_model(sd.generator.KDE()),
    'GMM': run_model(sd.generator.GMM()),
}
arguments = {k: dict() if k != 'Clasificador' else {'class_weight': 'balanced'} for k in treatments.keys()}
rates = [1, 5, 10, 50, 100, 500]
limit = 2500

results0 = {k: [0 for _ in rates] for k in treatments.keys()}
results1 = {k: [0 for _ in rates] for k in treatments.keys()}
for i in range(repeats):
    for j, pair in enumerate(np.random.choice(labels, (6, 2), False)):
        reduction = int(limit / rates[j])
        pool1 = ideal_reduced[ideal_reduced[target] == pair[0]].index
        pool2 = ideal_reduced[ideal_reduced[target] == pair[1]].index
        n1, n2 = len(pool1), len(pool2)
        ind1 = pool1.to_numpy()[np.random.choice(n1, (limit, 2), False)]
        ind2 = pool2.to_numpy()[np.random.choice(n2, (limit, 2), False)]
        train = ideal_reduced.iloc[np.concatenate([ind1[:reduction,0], ind2[:,0]])].copy().reset_index(drop=True)
        test = ideal_reduced.iloc[np.concatenate([ind1[:,1], ind2[:,1]])].copy().reset_index(drop=True)
        for k, v in treatments.items():
            treated_train = treatments[k](train)
            val0, val1 = analize_ideal(treated_train, test, pair, **arguments[k])
            results0[k][j] += val0 / repeats
            results1[k][j] += val1 / repeats
 
out0 = pd.DataFrame(results0)
out1 = pd.DataFrame(results1)
out0.to_csv("ignore/ideal_reduced_0.csv")
out1.to_csv("ignore/ideal_reduced_1.csv")
