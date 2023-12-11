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
housing = pd.read_csv("datasets/housing_dataset.csv")
dh = sd.DataHub()
dh.load(housing)
encoders = dh.encoders

def analize_housing(train, test):
    from sklearn import pipeline
    from sklearn import preprocessing
    from sklearn import compose
    from sklearn import neighbors
    from sklearn import metrics
    
    target = train.columns[0]
    data = train.columns != target
    
    encoded = set(['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning',
               'prefarea', 'furnishingstatus']).intersection(train.columns)
    steps = [('encoder', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), list(encoded))]
    pipe = pipeline.make_pipeline(
        compose.ColumnTransformer(steps, remainder='passthrough'),
        preprocessing.StandardScaler(),
        neighbors.KNeighborsRegressor()
    )
    pipe.fit(train.loc[:,data], train.loc[:,target])
    real = test.loc[:,target]
    pred = pipe.predict(test.loc[:,data])
    return metrics.mean_squared_error(real, pred)

repeats = 1000

def run_model(model):
    def func(df, _):
        dh_ = sd.DataHub(model=model)
        dh_.load(df, encoders=encoders)
        return (dh_.fill(), _)
    return func

lost = [col in ['area', 'hotwaterheating', 'airconditioning', 'furnishingstatus'] for col in housing.columns]
treatments = {
    'treure_mostres': lambda df, _: (df[~df.isna().any(1)], _),
    'treure_columnes': lambda df1, df2: (df1[df1.columns[~df1.isna().any(0)]], df2[df1.columns[~df1.isna().any(0)]]),
    'GMM': run_model(sd.generator.GMM()),
    'KDE': run_model(sd.generator.KDE()),
}

for percent in [0.1, 0.3, 0.5, 0.7]:
    results = {k: [] for k in treatments.keys()}
    for i in range(repeats):
        n, half = len(dh.data), int(len(dh.data) / 2)
        ind = np.random.choice(n, n, False)
        train = housing.iloc[ind[:half]].copy().reset_index(drop=True)
        test = housing.iloc[ind[half:]].copy().reset_index(drop=True)
        missing = int(percent * half)
        train.loc[0:missing, lost] = np.nan
        for k, v in treatments.items():
            treated_train, corrected_test = treatments[k](train, test)
            results[k].append(analize_housing(treated_train, corrected_test))
    
    output = pd.DataFrame(results)
    output.to_csv(f"ignore/housing_results_{percent}.csv", sep=';')