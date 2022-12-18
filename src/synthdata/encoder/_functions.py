# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:37:51 2022

@author: david
"""

import numpy as np

from ._classes import EncoderNone, EncoderDiscrete, EncoderIgnore, EncoderOHE, EncoderLimit, EncoderScale
    
def greater(value: 'float' = 0, include: 'bool' = True, influence: 'float' = 1):
    """
    Encoder for values greater than value.

    Parameters
    ----------
    value : 'float', optional
        Minimum value. The default is 0.
    include : 'bool', optional
        Include minimum in posible values. The default is True.
    influence : 'float', optional
        Range of improbable values near the minimum when not included. The default is 1.

    Returns
    -------
    Encoder.
    
    """
    if include:
        return EncoderLimit(lower=value, tails=False)
    else:
        return EncoderLimit(lower=value, tails=True, influence=influence)
    
def lower(value: 'float' = 0, include: 'bool' = True, influence: 'float' = 1):
    """
    Encoder for values greater than value.

    Parameters
    ----------
    value : 'float', optional
        Minimum value. The default is 0.
    include : 'bool', optional
        Include minimum in posible values. The default is True.
    influence : 'float', optional
        Range of improbable values near the minimum when not included. The default is 1.

    Returns
    -------
    Encoder.
    
    """
    if include:
        return EncoderLimit(upper=value, tails=False)
    else:
        return EncoderLimit(upper=value, tails=True, influence=influence)

def auto(data: 'DataFrame'):
    """
    Select Encoder based on dataframe column.

    Parameters
    ----------
    data : 'DataFrame'
        Data frame column.

    Returns
    -------
    Encoder.

    """
    if data.dtype == object:
        symbols = data.unique()
        n = len(data)
        s = len(symbols)
        if s == 1:
            return EncoderIgnore(symbols[0])
        elif s == 2:
            return EncoderScale(symbols)
        elif np.sqrt(n) > s and s < 100:
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