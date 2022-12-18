# -*- coding: utf-8 -*-
"""
DataHub

@author: david
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import dill
from time import time as gettime

import synthdata.encoder as enc
from synthdata.generator._auto import AutoSelection
from ._transformer import Transformer

class DataHub(Transformer):
    def __init__(self, cores: 'int | None' = 1, model: 'Generator' = AutoSelection(), **kwargs):
        super().__init__(**kwargs)
        if cores is None:
            cores = 1
        self.use_mp = (cores != 1)
        self.cores = cores
        self.model = model
    
    def load(self, data: 'DataFrame', encoders: 'dict[str, Encoder]' = dict()):
        assert len(data.shape) == 2, "data must be a 2d array of shape (n_samples, n_features)"
        
        self.data = data
        self.get_info(data)
        self.add_encoders({
            label: enc.auto(data[label])
            for label in self.labels
            } | encoders)
        self.samples, self.features = data.shape
        
    def set_model(self, model):
        self.model = model
        
    def set_cores(self, cores=None):
        if not cores:
            cores = 1
        self.use_mp = (cores != 1)
        self.cores = cores
    
    MP_DATA_FILE = 'temp.pickle'
    
    def pickle_data(self):
        class PickleManager:
            def __enter__(*_):
                self.data.to_pickle(DataHub.MP_DATA_FILE)
                return DataHub.MP_DATA_FILE
            
            def __exit__(*_):
                import os
                os.remove(DataHub.MP_DATA_FILE)
                return False
            
        return PickleManager()
    
    def _run_mp(what):
        FUN, trans, (target, key), args, kwds = dill.loads(what)
        data = pd.read_pickle(DataHub.MP_DATA_FILE)
        subdata = data[data[target] == key]
        return FUN(trans, subdata, *args, **kwds)
    
    def for_target(self, target, FUN, *args, **kwargs):
        if target is None:
            results = {'all': FUN(self, self.data, *args, **kwargs)}
        else:
            target_encoder = self.encoders[target]
            results = dict()
            if self.use_mp:
                keys = self.data[target].unique()
                values = []
                neccesary_cores = len(keys)
                allowed_cores = self.cores if self.cores > 0 else (mp.cpu_count() + self.cores)
                cores = min(neccesary_cores, allowed_cores)
                with self.pickle_data(), mp.Pool(cores) as pool:
                    for key in keys:
                        self.encoders[target] = enc.ignore(default=key)
                        trans = self.copy_transformer()
                        what = [dill.dumps((FUN, trans, (target, key), args, kwargs))]
                        values.append(pool.apply_async(DataHub._run_mp, what))
                    for key, value in zip(keys, values):
                        results[str(key)] = value.get()
            else:
                for value in self.data[target].unique():
                    self.encoders[target] = enc.ignore(default=value)
                    subdata = self.data[self.data[target] == value]
                    results[str(value)] = FUN(self, subdata, *args, **kwargs)
            self.encoders[target] = target_encoder
        return results
    
    def _kfold(trans, subdata, model, train_samples, validation_samples, folds, validation, return_fit, return_time):
        subsample = len(subdata)
        fold_size = subsample // folds
        total = folds * fold_size
        train_samples = total - fold_size if train_samples is None else train_samples
        validation_samples = fold_size if validation_samples is None else validation_samples
        
        sampling = np.reshape(np.random.choice(subsample, total, False), (folds, -1))
        value = 0
        selfvalue = 0
        fit_time = 0
        eval_time = 0
        for i in range(folds):
            ind = [(i + j) % folds for j in range(1, folds)]
            train = subdata.iloc[sampling[ind].flatten()[:train_samples]]
            test = subdata.iloc[sampling[i][:validation_samples]]
            start = gettime()
            trans.run(train, model.fit)
            fit_time += gettime() - start
            if validation == 'loglikelihood':
                if return_fit:
                    selfvalue += model.loglikelihood(trans.transform(train)) / train_samples + np.log(trans.deformation())
                start = gettime()
                value += model.loglikelihood(trans.transform(test)) / validation_samples + np.log(trans.deformation())
                eval_time += gettime() - start
            else:
                start = gettime()
                Xgen = trans.transform(trans.inv_transform(model.generate(validation_samples)), process=False)
                eval_time += gettime() - start
                Xtest = trans.transform(test, process=False)
                value += validation(Xtest, Xgen)
                if return_fit:
                    if train_samples > validation_samples:
                        Xgen = trans.transform(trans.inv_transform(model.generate(train_samples)), process=False)
                    elif train_samples < validation_samples:
                        Xgen = Xgen[:train_samples]
                    Xtrain = trans.transform(train, process=False)
                    selfvalue += validation(Xtrain, Xgen)
        output = {'validation': value / folds}
        if return_fit:
            output |= {'train': selfvalue / folds}
        if return_time:
            output |= {'fitting_time': fit_time / folds, 'evaluation_time': eval_time / folds}
        return output
    
    def kfold_validation(self, folds: 'int' = None,
                         train_samples: 'int | None' = None, validation_samples: 'int | None' = None,
                         validation: 'str | function' = 'loglikelihood',
                         model: 'Generator | None' = None, target: 'str | None' = None,
                         return_fit: 'bool' = False, return_time: 'bool' = True):
        model = self.model if model is None else model
        if folds is None:
            raise ValueError("No value specified for number of folds in kfold validation")
        args = {
            'model': model,
            'train_samples': train_samples,
            'validation_samples': validation_samples,
            'folds': folds,
            'validation': validation,
            'return_fit': return_fit,
            'return_time': return_time
        }
        if target is None:
            return self.for_target(target, DataHub._kfold, **args)['all']
        return self.for_target(target, DataHub._kfold, **args)

    def _generate(trans, subdata, model, n_samples):
        trans.run(subdata, model.fit)
        return trans.inv_transform(model.generate(n_samples))
    
    def generate(self, n_samples: 'int',
                 model: 'Generator | None' = None, target: 'str | None' = None):
        model = self.model if model is None else model
        
        output = self.for_target(target, DataHub._generate, model=model, n_samples=n_samples)
        return pd.concat(output.values(), ignore_index=True)
        
    def _fill(trans, subdata, model):
        new_data = subdata.copy()
        nans = trans.run(new_data, model.fit)
        new_data.iloc[nans] = trans.inv_transform(
            model.fill(trans.transform(new_data[nans]))
        )
        return new_data
    
    def fill(self, model: 'Generator | None' = None, target: 'str | None' = None):
        model = self.model if model is None else model
        
        output = self.for_target(target, DataHub._fill, model)
        return pd.concat(output.values(), ignore_index=True)
     
    def _extend(trans, subdata, model, n_samples, max_samples):
        subsample = len(subdata)
        if n_samples <= subsample:
            return subdata.iloc[np.random.choice(subsample, min(max_samples, subsample), False)]
        else:
            trans.run(subdata, model.fit)
            new_data = trans.inv_transform(model.generate(n_samples - subsample))
            return pd.concat([subdata, new_data])
    
    def extend(self, n_samples: 'str | int', max_samples: 'str | int' = 'n_samples',
               on_empty: 'str' = 'ignore',
               model: 'Generator | None' = None, target: 'str | None' = None):
        model = self.model if model is None else model
        if isinstance(n_samples, str):
            assert isinstance(target, str), "Special values for n_samples require a target column"
            if n_samples.lower() in ['min_target']:
                n_samples = min(np.unique(self.data[target].to_numpy(), return_counts=True)[1])
            elif n_samples.lower() in ['max_target']:
                n_samples = max(np.unique(self.data[target].to_numpy(), return_counts=True)[1])
            else:
                raise ValueError(f"Unrecognized special value '{n_samples}' for n_samples, use 'min_target' or 'max_target' instead")
        if max_samples.lower() in ['n_samples']:
            max_samples = n_samples
        elif max_samples.lower() in ['max']:
            max_samples = self.samples
        else:
            max_samples = max(max_samples, n_samples)
            
        output = self.for_target(target, DataHub._extend, model, n_samples, max_samples)
        return pd.concat(output.values(), ignore_index=True)
