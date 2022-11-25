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
from ._transformer import Transformer

class DataHub(Transformer):
    def __init__(self, cores=None, **kwrds):
        super().__init__(**kwrds)
        if not cores:
            cores = 1
        self.use_mp = (cores != 1)
        self.cores = cores
    
    def load(self, data, encoders=dict(), method=None):
        assert len(data.shape) == 2, "data must be a 2d array of shape (n_samples, n_features)"
        
        self.data = data
        self.get_info(data)
        self.add_encoders({
            label: enc.auto(data[label])
            for label in self.labels
            } | encoders)
        self.method = None
        self.samples, self.features = data.shape
        
    def set_method(self, method):
        self.method = method
        
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
                with mp.Pool(cores) as pool, self.pickle_data():
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
    
    def _kfold(trans, subdata, method, n_samples, folds, validation, return_fit, return_time):
        def sample_fold_total(n_samples, folds, total):
            if folds is None:
                assert total >= 2 * n_samples, "n_samples to big for the data"
                folds = total // n_samples
                return n_samples, folds, n_samples * folds
            elif n_samples is None:
                assert folds >= 2, "folds must be at least 2"
                n_samples = total // folds
                return n_samples, folds, folds * n_samples
            else:
                assert total >= 2 * n_samples, "n_samples to big for the data"
                folds = min(folds, total // n_samples)
                return n_samples, folds, n_samples * folds
        
        subsample = len(subdata)
        sample, fold, total = sample_fold_total(n_samples, folds, subsample)
        sampling = np.reshape(np.random.choice(subsample, total, False), (fold, -1))
        value = 0
        selfvalue = 0
        fit_time = 0
        eval_time = 0
        for i in range(fold):
            train = subdata.iloc[sampling[i]]
            test = subdata.iloc[sampling[(i + 1) % fold]]
            start = gettime()
            trans.run(train, method.fit)
            fit_time += gettime() - start
            if validation == 'loglikelihood':
                if return_fit:
                    selfvalue += method.loglikelihood(trans.transform(train)) / sample + np.log(trans.deformation())
                start = gettime()
                value += method.loglikelihood(trans.transform(test)) / sample + np.log(trans.deformation())
                eval_time += gettime() - start
            else:
                start = gettime()
                Xgen = trans.transform(trans.inv_transform(method.generate(sample)), process=False)
                eval_time += gettime() - start
                Xtrain = trans.transform(train, process=False)
                Xtest = trans.transform(test, process=False)
                if return_fit:
                    selfvalue += validation(Xtrain, Xgen)
                value += validation(Xtest, Xgen)
        output = {'validation': value / fold}
        if return_fit:
            output |= {'train': selfvalue / fold}
        if return_time:
            output |= {'fitting_time': fit_time / fold, 'evaluation_time': eval_time / fold}
        return output
    
    def kfold_validation(self, n_samples=None, folds=None, method=None, validation='loglikelihood', target=None, return_fit=False, return_time=True):
        method = self.method if method is None else method
        if n_samples is None and folds is None:
            raise ValueError("No value specified for kfold validation")
        args = {
            'method': method,
            'n_samples': n_samples,
            'folds': folds,
            'validation': validation,
            'return_fit': return_fit,
            'return_time': return_time
        }
        if target is None:
            return self.for_target(target, DataHub._kfold, **args)['all']
        return self.for_target(target, DataHub._kfold, **args)

    def _generate(trans, subdata, method, n_samples):
        trans.run(subdata, method.fit)
        return trans.inv_transform(method.generate(n_samples))
    
    def generate(self, n_samples, method=None, target=None):
        method = self.method if method is None else method
        
        output = self.for_target(target, DataHub._generate, method=method, n_samples=n_samples)
        return pd.concat(output.values(), ignore_index=True)
        
    def _fill(trans, subdata, method):
        new_data = subdata.copy()
        nans = trans.run(new_data, method.fit)
        new_data.iloc[nans] = trans.inv_transform(
            method.fill(trans.transform(new_data[nans]))
        )
        return new_data
    
    def fill(self, method=None, target=None):
        method = self.method if method is None else method
        
        output = self.for_target(target, DataHub._fill, method)
        return pd.concat(output.values(), ignore_index=True)
     
    def _extend(trans, subdata, method, n_samples, max_samples):
        subsample = len(subdata)
        if n_samples <= subsample:
            return subdata.iloc[np.random.choice(subsample, min(max_samples, subsample), False)]
        else:
            trans.run(subdata, method.fit)
            new_data = trans.inv_transform(method.generate(n_samples - subsample))
            return pd.concat([subdata, new_data])
    
    def extend(self, n_samples, max_samples='n_samples', method=None, target=None):
        method = self.method if method is None else method
        if max_samples == 'n_samples':
            max_samples = n_samples
        elif max_samples is None:
            max_samples = self.samples
        else:
            max_samples = max(max_samples, n_samples)
            
        output = self.for_target(target, DataHub._extend, method, n_samples, max_samples)
        return pd.concat(output.values(), ignore_index=True)
