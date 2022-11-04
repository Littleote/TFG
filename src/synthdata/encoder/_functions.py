# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:37:51 2022

@author: david
"""

import numpy as np

from ._classes import EncoderNone, EncoderDiscrete, EncoderIgnore, EncoderOHE, EncoderLimit
    
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

def auto(data):
    if data.dtype == object:
        symbols = data.unique()
        use = np.sqrt(data.shape[0]) / symbols.shape[0]
        if use > 1:
            return EncoderOHE(list(symbols))
        else:
            return EncoderIgnore('Ignored')
    else:
        if data.dtype == np.float64:
            if (data > 1e-6).all():
                return EncoderLimit(lower=0, tails=True)
            elif (data >= -0).all():
                return EncoderLimit(lower=0, tails=False)
            else:
                return EncoderNone()
        else:
            minimum = None
            if (data > 0).all():
                minimum = 1
            elif (data >= 0).all():
                minimum = 0
            return EncoderDiscrete(minimum=minimum)