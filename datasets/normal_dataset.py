# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import pandas as pd

# Sphere dataset
def normal(N, dimension):
    noise = np.random.normal(size=(N, dimension))
    norm = np.linalg.norm(noise, axis=1)
    
    sphere = pd.DataFrame(noise, columns=[f'dimension_{i + 1}' for i in range(dimension)])
    sphere['zone'] = ['inner' if r < 1 else 'outter' for r in norm]
    return sphere
    
if __name__ == "__main__":
    print(normal(10, 4))