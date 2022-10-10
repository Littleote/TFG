# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:37:51 2022

@author: david
"""

import numpy as np

from ._classes import EncoderNone, EncoderIgnore, EncoderOHE, EncoderLimit

def auto(data):
    if data.dtype == object:
        symbols = data.unique()
        use = np.sqrt(data.shape[0]) / symbols.shape[0]
        if use > 1:
            return EncoderOHE(list(symbols))
        else:
            return EncoderIgnore('Ignored')
    else:
        return EncoderNone()
    
def greater(value=0, include=True, influence=1):
    if include:
        return EncoderLimit(lower=value, tails=False)
    else:
        return EncoderLimit(lower=value, tails=True, influence=influence)
    
def lower(value=0, include=True, influence=1):
    if include:
        return EncoderLimit(upper=value, tails=False)
    else:
        return EncoderLimit(upper=value, tails=True, influence=influence)